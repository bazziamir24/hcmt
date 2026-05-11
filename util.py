import enum
import itertools
import re
import tensorflow as tf

class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SYMMETRIC = 7
    SIZE = 9


def _unique_edges(edges):
    """Returns a two-way edge list from an undirected edge array."""
    # Those edges are sometimes duplicated within the mesh and sometimes
    # single at the mesh boundary.
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)

    # remove duplicates and unpack
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)

    # create two-way connectivity
    return tf.concat([senders, receivers], axis=0), tf.concat([receivers, senders], axis=0)


def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    edges = tf.concat(
        [
            faces[:, 0:2],
            faces[:, 1:3],
            tf.stack([faces[:, 2], faces[:, 0]], axis=1),
        ],
        axis=0,
    )
    return _unique_edges(edges)


def tetrahedra_to_edges(cells):
    """Computes mesh edges from tetrahedra."""
    edge_pairs = list(itertools.combinations(range(4), 2))
    edges = tf.concat(
        [tf.gather(cells, pair, axis=1) for pair in edge_pairs],
        axis=0,
    )
    return _unique_edges(edges)


def cells_to_edges(cells):
    """Computes mesh edges from triangle or tetrahedral cells."""
    cell_size = cells.shape[-1]

    if cell_size == 3:
        return triangles_to_edges(cells)
    if cell_size == 4:
        return tetrahedra_to_edges(cells)

    raise ValueError(
        f"Unsupported cell arity {cell_size}. Expected triangles (3) or tetrahedra (4)."
    )


def get_check_point_num(path):
    try:
        file = open(path, 'r')
        data = file.readlines()
        file.close()
        return re.sub(r'[^0-9]', '', data[0])
    except:
        return 0


def get_mask_impact_normal(initial_state):
    mask_x = tf.logical_or(
        tf.equal(initial_state['node_type'][:, 0], NodeType.HANDLE),
        tf.equal(initial_state['node_type'][:, 0], NodeType.SYMMETRIC)
    )
    mask_y = tf.equal(initial_state['node_type'][:, 0], NodeType.HANDLE)
    mask = tf.stack([mask_x, mask_y], axis=1)
    mask = tf.logical_not(mask)
    return mask


def get_mask_deforming_plate(initial_state):
    """Allows only NORMAL nodes to move in all spatial dimensions."""
    movable = tf.equal(initial_state['node_type'][:, 0], NodeType.NORMAL)
    dim = tf.shape(initial_state['world_pos'])[1]
    return tf.tile(movable[:, None], [1, dim])
