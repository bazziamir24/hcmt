import argparse
from pathlib import Path

import h5py
import numpy as np


NUM_HIERARCHY_LEVELS = 7


def pairwise_squared_distance(points):
    diff = points[:, None, :] - points[None, :, :]
    return np.sum(diff * diff, axis=-1)


def farthest_point_sampling(points, target_count):
    """Returns indices of a farthest-point sample."""
    num_points = points.shape[0]
    if target_count >= num_points:
        return np.arange(num_points, dtype=np.int32)

    selected = np.empty(target_count, dtype=np.int32)
    selected[0] = 0

    min_dist = np.sum((points - points[0]) ** 2, axis=1)
    min_dist[0] = -1.0

    for i in range(1, target_count):
        next_idx = int(np.argmax(min_dist))
        selected[i] = next_idx
        candidate_dist = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_dist = np.minimum(min_dist, candidate_dist)
        min_dist[selected[: i + 1]] = -1.0

    return np.sort(selected)


def knn_edges(points, k):
    """Builds a symmetric k-NN graph on the given points."""
    num_points = points.shape[0]
    if num_points <= 1:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    k = min(k, num_points - 1)
    dist = pairwise_squared_distance(points)
    np.fill_diagonal(dist, np.inf)

    neighbors = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
    undirected = set()
    for src in range(num_points):
        for dst in neighbors[src]:
            a, b = sorted((src, int(dst)))
            undirected.add((a, b))

    undirected = np.array(sorted(undirected), dtype=np.int32)
    senders = np.concatenate([undirected[:, 0], undirected[:, 1]], axis=0)
    receivers = np.concatenate([undirected[:, 1], undirected[:, 0]], axis=0)
    return senders, receivers


def next_level_count(num_nodes):
    if num_nodes <= 1:
        return 1
    return max(1, num_nodes // 2)


def build_hierarchy(mesh_pos, levels=NUM_HIERARCHY_LEVELS, knn_k=8):
    """Builds HCMT hierarchy arrays for a single mesh."""
    current_points = np.asarray(mesh_pos, dtype=np.float32)
    level_ids = []
    level_senders = []
    level_receivers = []

    for _ in range(levels):
        target_count = next_level_count(current_points.shape[0])
        idx = farthest_point_sampling(current_points, target_count)
        coarse_points = current_points[idx]
        senders, receivers = knn_edges(coarse_points, knn_k)

        level_ids.append(idx.astype(np.int32))
        level_senders.append(senders.astype(np.int32))
        level_receivers.append(receivers.astype(np.int32))

        current_points = coarse_points

    return level_ids, level_senders, level_receivers


def write_varlen_group(group, name, arrays):
    subgroup = group.require_group(name)
    for child in list(subgroup.keys()):
        del subgroup[child]
    for i, array in enumerate(arrays):
        subgroup.create_dataset(str(i), data=array, compression="gzip")


def process_file(input_path, output_path, levels, knn_k):
    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        for traj_name in src.keys():
            src_group = src[traj_name]
            dst_group = dst.create_group(traj_name)

            for dataset_name in src_group.keys():
                src.copy(src_group[dataset_name], dst_group, name=dataset_name)

            mesh_pos = src_group["mesh_pos"][0]
            m_ids, m_gs_s, m_gs_r = build_hierarchy(mesh_pos, levels=levels, knn_k=knn_k)
            write_varlen_group(dst_group, "m_ids", m_ids)
            write_varlen_group(dst_group, "m_gs_s", m_gs_s)
            write_varlen_group(dst_group, "m_gs_r", m_gs_r)

            print(
                f"processed trajectory {traj_name}: "
                f"nodes={mesh_pos.shape[0]}, "
                f"coarsest={m_ids[-1].shape[0]}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Generate HCMT hierarchy tensors from HDF5 mesh trajectories."
    )
    parser.add_argument("input_h5", type=Path, help="Path to the source HDF5 file.")
    parser.add_argument(
        "output_h5",
        type=Path,
        help="Path to the output HDF5 file with hierarchy datasets added.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        default=NUM_HIERARCHY_LEVELS,
        help="Number of hierarchy levels to generate.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=8,
        help="Number of nearest neighbors used for each coarse graph.",
    )
    args = parser.parse_args()

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    process_file(args.input_h5, args.output_h5, levels=args.levels, knn_k=args.knn_k)


if __name__ == "__main__":
    main()
