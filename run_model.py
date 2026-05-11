import tensorflow as tf
import numpy as np
import os
import dataset
import random
import pickle

from pathlib import Path
from util import get_check_point_num
from termcolor import colored
from tqdm import tqdm
from core_model import HCMT
from model import PlateModel, evaluate_plate

from absl import app
from absl import flags
from absl import logging

print(colored(f'tensorflow version : {tf.__version__}', 'red'))
print(colored(f'GPUs Available : {len(tf.config.experimental.list_physical_devices("GPU"))}', 'red'))

gpus = tf.config.list_physical_devices('GPU')

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'eval', 'predict'], 'Train model, run evaluation, or export prediction VTKs.')
flags.DEFINE_string('dataset_dir', "datasets", 'Directory containing train.h5/test.h5 files.')
flags.DEFINE_string('checkpoint_dir', 'workspace/run/check', 'Directory to save checkpoint')
flags.DEFINE_string('rollout_dir', 'workspace/run/rollout', 'Pickle file to save eval trajectories')
flags.DEFINE_string('logging_dir', 'workspace/run/log', 'log directory')
flags.DEFINE_string('vtk_dir', 'workspace/run/vtk', 'Directory to save VTK prediction files')
flags.DEFINE_integer('hierarchy_levels', 2, 'Number of hierarchy levels stored in the dataset.')
flags.DEFINE_bool('deterministic_ops', False, 'Enable deterministic TensorFlow ops. Disable for GPU training with unsorted_segment_sum.')
flags.DEFINE_integer('num_training_steps', 1000000, 'No. of training steps')
flags.DEFINE_integer('num_rollouts', 20, 'No. of rollouts')
flags.DEFINE_integer('num_vtk_rollouts', 3, 'Number of test trajectories to export as VTK in predict mode.')
flags.DEFINE_integer('seed', 42, 'No. of random seed')

for i in range(len(gpus)):
	tf.config.experimental.set_memory_growth(gpus[i], True)


def learner(model):

    @tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            loss = model.loss(inputs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    ds = dataset.load_plate_frame_dataset(FLAGS.dataset_dir, 'train')
    ds = ds.shuffle(2500, seed=FLAGS.seed, reshuffle_each_iteration=False)
    ds = ds.repeat(None).prefetch(10)

    ds = tf.compat.v1.data.make_one_shot_iterator(ds)

    global_step = tf.Variable(0, name='global_step', trainable=False)
    ckpt = tf.train.Checkpoint(step=global_step, net=model)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=FLAGS.checkpoint_dir, max_to_keep=50)
    ckpt.restore(manager.latest_checkpoint)

    lr_schedule = tf.compat.v1.train.exponential_decay(learning_rate=1e-4,
                                global_step=global_step,
                                decay_steps=int(1000000),
                                decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    losses = 0.0

    """ Training """
    counter = 0
    epoch_steps = 100000
    total_steps = FLAGS.num_training_steps - int(global_step) + 1
    total_progress = tqdm(total=total_steps, desc='Training', unit='step')
    epoch_progress = tqdm(total=epoch_steps, desc='Epoch', unit='step', leave=False)
    for step in range(int(global_step), FLAGS.num_training_steps + 1, 1):
        inputs = ds.get_next()

        if step < 1000:
            model._build_graph(inputs, True)
            total_progress.update(1)
            epoch_progress.update(1)
        else:
            loss = train_step(inputs)
            loss_value = float(loss.numpy())
            losses += loss_value
            counter += 1
            avg_loss = losses / counter

            total_progress.update(1)
            epoch_progress.update(1)
            total_progress.set_postfix(loss=f'{loss_value:.6f}', avg=f'{avg_loss:.6f}')
            epoch_progress.set_postfix(loss=f'{loss_value:.6f}', avg=f'{avg_loss:.6f}')

            if counter != 1 and step % epoch_steps == 0:
                manager.save(checkpoint_number=int(global_step))
                print(f'{step} {avg_loss}')
            
            if counter != 1 and step % int(epoch_steps/200) == 0:
                print(f'{step} {avg_loss:.9f}')

        if epoch_progress.n >= epoch_steps:
            epoch_progress.close()
            epoch_progress = tqdm(total=epoch_steps, desc='Epoch', unit='step', leave=False)

        global_step.assign_add(1)

    total_progress.close()
    epoch_progress.close()
    manager.save(checkpoint_number=int(global_step))
    with open(os.path.join(FLAGS.logging_dir, 'train_epoch_RMSE.txt'), 'a') as file:
        file.write(f'{step} {losses/counter}\n')


def _restore_checkpoint(model):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    ckpt = tf.train.Checkpoint(step=global_step, net=model)
    manager = tf.train.CheckpointManager(checkpoint=ckpt, directory=FLAGS.checkpoint_dir, max_to_keep=None)
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    checkpoint_num = get_check_point_num(os.path.join(FLAGS.checkpoint_dir, 'checkpoint'))
    return checkpoint_num


def _to_numpy_tree(tree):
    return {
        key: value.numpy() if hasattr(value, 'numpy') else value
        for key, value in tree.items()
    }


def _vtk_cell_type(cell_width):
    if cell_width == 3:
        return 5
    if cell_width == 4:
        return 10
    raise ValueError(f'Unsupported VTK cell width: {cell_width}')


def _write_legacy_vtk(path, points, cells, node_type, gt_points):
    points = np.asarray(points, dtype=np.float32)
    cells = np.asarray(cells, dtype=np.int32)
    node_type = np.asarray(node_type, dtype=np.int32).reshape(-1)
    gt_points = np.asarray(gt_points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'Expected points with shape [N, 3], got {points.shape}')
    if cells.ndim != 2:
        raise ValueError(f'Expected cells with shape [M, K], got {cells.shape}')

    pred_displacement = points - gt_points
    cell_width = cells.shape[1]
    cell_type = _vtk_cell_type(cell_width)

    with open(path, 'w', encoding='ascii') as fp:
        fp.write('# vtk DataFile Version 3.0\n')
        fp.write('HCMT predicted rollout\n')
        fp.write('ASCII\n')
        fp.write('DATASET UNSTRUCTURED_GRID\n')
        fp.write(f'POINTS {points.shape[0]} float\n')
        for point in points:
            fp.write(f'{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n')

        num_cells = cells.shape[0]
        vtk_cells_size = num_cells * (cell_width + 1)
        fp.write(f'CELLS {num_cells} {vtk_cells_size}\n')
        for cell in cells:
            cell_entries = ' '.join(str(int(index)) for index in cell)
            fp.write(f'{cell_width} {cell_entries}\n')

        fp.write(f'CELL_TYPES {num_cells}\n')
        for _ in range(num_cells):
            fp.write(f'{cell_type}\n')

        fp.write(f'POINT_DATA {points.shape[0]}\n')
        fp.write('SCALARS node_type int 1\n')
        fp.write('LOOKUP_TABLE default\n')
        for value in node_type:
            fp.write(f'{int(value)}\n')

        fp.write('VECTORS ground_truth_position float\n')
        for point in gt_points:
            fp.write(f'{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n')

        fp.write('VECTORS prediction_displacement float\n')
        for vector in pred_displacement:
            fp.write(f'{vector[0]:.9g} {vector[1]:.9g} {vector[2]:.9g}\n')

        fp.write('SCALARS prediction_error float 1\n')
        fp.write('LOOKUP_TABLE default\n')
        for error in np.linalg.norm(pred_displacement, axis=1):
            fp.write(f'{float(error):.9g}\n')


def _export_rollout_vtks(checkpoint_num, traj_idx, traj_data):
    vtk_root = Path(FLAGS.vtk_dir) / str(checkpoint_num) / f'traj_{traj_idx:03d}'
    vtk_root.mkdir(parents=True, exist_ok=True)

    np_traj = _to_numpy_tree(traj_data)
    num_steps = np_traj['pred_pos'].shape[0]
    for step_idx in range(num_steps):
        vtk_path = vtk_root / f'frame_{step_idx:04d}.vtk'
        _write_legacy_vtk(
            vtk_path,
            points=np_traj['pred_pos'][step_idx],
            cells=np_traj['cells'][step_idx],
            node_type=np_traj['node_type'][step_idx],
            gt_points=np_traj['gt_pos'][step_idx],
        )


def evaluator(model):

    ds = dataset.load_plate_trajectory_dataset(FLAGS.dataset_dir, 'test')
    ds = tf.compat.v1.data.make_one_shot_iterator(ds)

    trajectories = []
    scalars = []

    checkpoint_num = _restore_checkpoint(model)

    print(colored(checkpoint_num, 'red'))
    counter = 0
    for traj_idx in range(FLAGS.num_rollouts):
        inputs = ds.get_next()
        scalar_data, traj_data = evaluate_plate(model, inputs)
        trajectories.append(traj_data)

        scalars.append(scalar_data)
        print(traj_idx, scalar_data)
        counter += 1
        del traj_data
        del inputs

    with open(os.path.join(FLAGS.logging_dir, 'test_RMSE.txt'), 'a') as file:
        txt = ''
        for key in scalars[0]:
            print('%s: %g', key, np.mean([x[key] for x in scalars]))
            txt += f' {key} {np.mean([x[key] for x in scalars])}'
        file.write(f'{checkpoint_num} {txt}\n')

    with open(os.path.join(FLAGS.rollout_dir, f'{checkpoint_num}.pkl'), 'wb') as fp:
        pickle.dump(trajectories, fp)


def predictor(model):
    ds = dataset.load_plate_trajectory_dataset(FLAGS.dataset_dir, 'test')
    ds = tf.compat.v1.data.make_one_shot_iterator(ds)

    checkpoint_num = _restore_checkpoint(model)
    print(colored(checkpoint_num, 'red'))

    num_exports = max(0, FLAGS.num_vtk_rollouts)
    for traj_idx in range(num_exports):
        inputs = ds.get_next()
        _, traj_data = evaluate_plate(model, inputs)
        _export_rollout_vtks(checkpoint_num, traj_idx, traj_data)
        print(f'Exported VTK rollout {traj_idx} to {os.path.join(FLAGS.vtk_dir, str(checkpoint_num), f"traj_{traj_idx:03d}")}')


def main(argv):
    del argv

    tf.compat.v1.enable_resource_variables()
    tf.config.run_functions_eagerly(False)

    """ Create base directory """
    Path(FLAGS.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.rollout_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.logging_dir).mkdir(parents=True, exist_ok=True)
    Path(FLAGS.vtk_dir).mkdir(parents=True, exist_ok=True)

    """ Fix seed """
    tf.keras.utils.set_random_seed(FLAGS.seed)
    if FLAGS.deterministic_ops:
        tf.config.experimental.enable_op_determinism()
    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)
    tf.compat.v1.set_random_seed(FLAGS.seed)

    model = PlateModel(HCMT(hierarchy_levels=FLAGS.hierarchy_levels + 1))

    if FLAGS.mode == 'train':
        learner(model)

    if FLAGS.mode == 'eval':
        evaluator(model)

    if FLAGS.mode == 'predict':
        predictor(model)


if __name__ == '__main__':
    app.run(main)
