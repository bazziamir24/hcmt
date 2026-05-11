import functools
import json
import os
import h5py
import numpy as np
import tensorflow.compat.v1 as tf
from util import NodeType, cells_to_edges


def _parse(proto, meta):
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta['field_names']}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta['features'].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field['dtype']))
        data = tf.reshape(data, field['shape'])
        if field['type'] == 'static':
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
            out[key] = data
        elif field['type'] == 'static_varlen':
            length = tf.io.decode_raw(features['length_' + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_splits(data, length)
   
            data = tf.expand_dims(data, 0)
       
            data = tf.tile(data, [meta['trajectory_length'], 1, 1])
            out[key] = data
        elif field['type'] == 'dynamic':
            out[key] = data
        elif field['type'] != 'dynamic':
            raise ValueError('invalid data format')
    return out


def load_dataset(path, split):
    """Load dataset."""
    h5_path = os.path.join(path, split + '.h5')
    if os.path.exists(h5_path):
        raise ValueError(
            'HDF5 datasets require load_plate_frame_dataset/load_plate_trajectory_dataset.'
        )

    with open(os.path.join(path, 'meta.json'), 'r') as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split+'.tfrecord'))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=1)
    ds = ds.prefetch(1)
    return ds


def add_targets(ds, fields, add_history):
    """Adds target and optionally history fields to dataframe."""
    def fn(trajectory):
        out = {}
        for key, val in trajectory.items():
            out[key] = val[1:-1]
            if key in fields:
                if add_history:
                    out['prev|' + key] = val[0:-2]
                out['target|' + key] = val[2:]

        return out

    return ds.map(fn, num_parallel_calls=1)


def split_and_preprocess(ds, noise_field, noise_scale, noise_gamma, seed):
    
    """Splits trajectories into frames, and adds training noise."""
    def add_noise(frame):
        noise = tf.random.normal(tf.shape(frame[noise_field]), stddev=noise_scale, dtype=tf.float32, seed=seed)
        # don't apply noise to boundary nodes
        mask = tf.equal(frame['node_type'], NodeType.NORMAL)[:, 0]
        noise = tf.where(mask, noise, tf.zeros_like(noise))
        frame[noise_field] += noise
        frame['target|'+noise_field] += (1.0 - noise_gamma) * noise
        return frame

    ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
    ds = ds.map(add_noise, num_parallel_calls=1)
    ds = ds.shuffle(2500, seed=seed, reshuffle_each_iteration=False)
    ds = ds.repeat(None)

    return ds.prefetch(10)


def _plate_h5_path(path, split):
    return os.path.join(path, split + '.h5')


def _read_hierarchy(group, name):
    return [group[name][str(i)][:].astype(np.int32) for i in range(len(group[name]))]


def _with_identity_level(cells, mesh_pos, m_ids, m_gs_s, m_gs_r):
    """Prepends the full-resolution mesh as hierarchy level 0."""
    num_nodes = mesh_pos.shape[0]
    identity_idx = np.arange(num_nodes, dtype=np.int32)
    senders, receivers = cells_to_edges(tf.convert_to_tensor(cells, dtype=tf.int32))
    base_senders = senders.numpy().astype(np.int32)
    base_receivers = receivers.numpy().astype(np.int32)

    return (
        [identity_idx] + m_ids,
        [base_senders] + m_gs_s,
        [base_receivers] + m_gs_r,
    )


def load_plate_frame_dataset(path, split):
    """Loads training frames from an HDF5 deforming-plate dataset."""
    h5_path = _plate_h5_path(path, split)

    def generator():
        with h5py.File(h5_path, 'r') as f:
            for traj_name in f.keys():
                group = f[traj_name]
                raw_m_ids = _read_hierarchy(group, 'm_ids')
                raw_m_gs_s = _read_hierarchy(group, 'm_gs_s')
                raw_m_gs_r = _read_hierarchy(group, 'm_gs_r')
                full_m_ids, full_m_gs_s, full_m_gs_r = _with_identity_level(
                    group['cells'][0],
                    group['mesh_pos'][0],
                    raw_m_ids,
                    raw_m_gs_s,
                    raw_m_gs_r,
                )
                hierarchy = {
                    'm_ids': tf.ragged.constant(
                        full_m_ids,
                        dtype=tf.int32,
                        row_splits_dtype=tf.int32,
                    ),
                    'm_gs_s': tf.ragged.constant(
                        full_m_gs_s,
                        dtype=tf.int32,
                        row_splits_dtype=tf.int32,
                    ),
                    'm_gs_r': tf.ragged.constant(
                        full_m_gs_r,
                        dtype=tf.int32,
                        row_splits_dtype=tf.int32,
                    ),
                }

                cells = group['cells'][:]
                mesh_pos = group['mesh_pos'][:]
                node_type = group['node_type'][:]
                world_pos = group['world_pos'][:]

                for t in range(1, world_pos.shape[0] - 1):
                    yield {
                        'cells': cells[t].astype(np.int32),
                        'mesh_pos': mesh_pos[t].astype(np.float32),
                        'node_type': node_type[t].astype(np.int32),
                        'prev|world_pos': world_pos[t - 1].astype(np.float32),
                        'world_pos': world_pos[t].astype(np.float32),
                        'target|world_pos': world_pos[t + 1].astype(np.float32),
                        **hierarchy,
                    }

    signature = {
        'cells': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
        'mesh_pos': tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        'node_type': tf.TensorSpec(shape=[None, 1], dtype=tf.int32),
        'prev|world_pos': tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        'world_pos': tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        'target|world_pos': tf.TensorSpec(shape=[None, 3], dtype=tf.float32),
        'm_ids': tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
        'm_gs_s': tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
        'm_gs_r': tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
    }

    return tf.data.Dataset.from_generator(generator, output_signature=signature)


def load_plate_trajectory_dataset(path, split):
    """Loads full rollouts from an HDF5 deforming-plate dataset."""
    h5_path = _plate_h5_path(path, split)

    def generator():
        with h5py.File(h5_path, 'r') as f:
            for traj_name in f.keys():
                group = f[traj_name]
                raw_m_ids = _read_hierarchy(group, 'm_ids')
                raw_m_gs_s = _read_hierarchy(group, 'm_gs_s')
                raw_m_gs_r = _read_hierarchy(group, 'm_gs_r')
                full_m_ids, full_m_gs_s, full_m_gs_r = _with_identity_level(
                    group['cells'][0],
                    group['mesh_pos'][0],
                    raw_m_ids,
                    raw_m_gs_s,
                    raw_m_gs_r,
                )
                yield {
                    'cells': group['cells'][1:-1].astype(np.int32),
                    'mesh_pos': group['mesh_pos'][1:-1].astype(np.float32),
                    'node_type': group['node_type'][1:-1].astype(np.int32),
                    'prev|world_pos': group['world_pos'][:-2].astype(np.float32),
                    'world_pos': group['world_pos'][1:-1].astype(np.float32),
                    'm_ids': tf.ragged.constant(
                        full_m_ids,
                        dtype=tf.int32,
                        row_splits_dtype=tf.int32,
                    ),
                    'm_gs_s': tf.ragged.constant(
                        full_m_gs_s,
                        dtype=tf.int32,
                        row_splits_dtype=tf.int32,
                    ),
                    'm_gs_r': tf.ragged.constant(
                        full_m_gs_r,
                        dtype=tf.int32,
                        row_splits_dtype=tf.int32,
                    ),
                }

    signature = {
        'cells': tf.TensorSpec(shape=[None, None, None], dtype=tf.int32),
        'mesh_pos': tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        'node_type': tf.TensorSpec(shape=[None, None, 1], dtype=tf.int32),
        'prev|world_pos': tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        'world_pos': tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
        'm_ids': tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
        'm_gs_s': tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
        'm_gs_r': tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int32, ragged_rank=1, row_splits_dtype=tf.int32),
    }

    return tf.data.Dataset.from_generator(generator, output_signature=signature).prefetch(1)

