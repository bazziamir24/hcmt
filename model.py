import sonnet as snt
import tensorflow as tf
import collections

from util import NodeType, cells_to_edges, get_mask_deforming_plate
from normalization import Normalizer


EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('Graph', ['node_features', 'edge_sets'])


class PlateModel(snt.Module):
    def __init__(self, learned_model, name='Model'):
        super(PlateModel, self).__init__(name=name)
        self._learned_model = learned_model

        # Normalizer
        self._output_normalizer = Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = Normalizer(size=3 + 3 + NodeType.SIZE, name='node_normalizer')


    def _get_obstacle_displacement(self, inputs):
        if 'obstacle_next_world_pos' in inputs:
            target_world_pos = inputs['obstacle_next_world_pos']
        elif 'target|world_pos' in inputs:
            target_world_pos = inputs['target|world_pos']
        else:
            raise KeyError(
                "Expected 'obstacle_next_world_pos' or 'target|world_pos' in inputs "
                "to build obstacle-motion features."
            )

        world_pos = inputs['world_pos']
        node_type = inputs['node_type'][:, 0]
        obstacle_mask = tf.equal(node_type, NodeType.OBSTACLE)

        obstacle_displacement = target_world_pos - world_pos
        obstacle_only = tf.boolean_mask(obstacle_displacement, obstacle_mask)

        mean_obstacle_displacement = tf.cond(
            tf.shape(obstacle_only)[0] > 0,
            lambda: tf.reduce_mean(obstacle_only, axis=0),
            lambda: tf.zeros([tf.shape(obstacle_displacement)[1]], dtype=obstacle_displacement.dtype),
        )

        return tf.where(
            obstacle_mask[:, None],
            obstacle_displacement,
            tf.broadcast_to(mean_obstacle_displacement[None, :], tf.shape(obstacle_displacement)),
        )


    def _build_graph(self, inputs, is_training):
        velocity = (inputs['world_pos'] - inputs['prev|world_pos'])
        obstacle_displacement = self._get_obstacle_displacement(inputs)
        node_type = tf.one_hot(inputs['node_type'][:, 0], NodeType.SIZE)

        node_features = tf.concat([velocity, obstacle_displacement, node_type], axis=-1)

        senders, receivers = cells_to_edges(inputs['cells'])

        edges = {}
        edges['m_senders'] = senders
        edges['m_receivers'] = receivers
        edges['c_senders'] = senders
        edges['c_receivers'] = receivers
 
        return self._node_normalizer(node_features, is_training), inputs, edges
        

    def __call__(self, inputs):
        node_features, inputs, edges = self._build_graph(inputs, is_training=False)
        per_node_network_output = self._learned_model(node_features, inputs, edges, is_training=False)

        return self._update(inputs, per_node_network_output)
    

    def loss(self, inputs):
        node_features, inputs, edges = self._build_graph(inputs, is_training=True)
        network_output = self._learned_model(node_features, inputs, edges, is_training=True)

        target_velocity = inputs['target|world_pos'] - inputs['world_pos']
        target_normalized = self._output_normalizer(target_velocity)

        loss_mask = get_mask_deforming_plate(inputs)
        error_vel = (target_normalized - network_output[:, :3]) ** 2
        error_vel = tf.where(loss_mask, error_vel, tf.zeros_like(error_vel))
        error_vel = tf.reduce_sum(error_vel, axis=-1)
        return tf.reduce_mean(error_vel)
    

    def _update(self, inputs, per_node_network_output):
        pred_velocity = self._output_normalizer.inverse(per_node_network_output[:, :3])
        position = inputs['world_pos'] + pred_velocity
        return position


def evaluate_plate(model, inputs):
    temporal_keys = {'cells', 'mesh_pos', 'node_type', 'prev|world_pos', 'world_pos'}

    def _rollout(model, initial_state, num_steps):
        mask = get_mask_deforming_plate(initial_state)
        obstacle_mask = tf.equal(initial_state['node_type'][:, 0], NodeType.OBSTACLE)
        obstacle_next_world_pos = tf.concat(
            [inputs['world_pos'][1:], inputs['world_pos'][-1:]],
            axis=0,
        )

        def step_fn(step, prev_pos, cur_pos, trajectory):
            prediction = model({
                **initial_state,
                'prev|world_pos': prev_pos,
                'world_pos': cur_pos,
                'obstacle_next_world_pos': obstacle_next_world_pos[step],
            })
            next_pos = tf.where(mask, prediction, cur_pos)
            next_pos = tf.where(obstacle_mask[:, None], obstacle_next_world_pos[step], next_pos)
            trajectory = trajectory.write(step, cur_pos)
            return step + 1, cur_pos, next_pos, trajectory
        
        _, _, _, output = tf.while_loop(
            cond=lambda step, prev, cur, traj: tf.less(step, num_steps),
            body=step_fn,
            loop_vars=(
                0,
                initial_state['prev|world_pos'],
                initial_state['world_pos'],
                tf.TensorArray(tf.float32, num_steps),
            ),
            parallel_iterations=1,
        )
        return output.stack()

    initial_state = {
        k: (v[0] if k in temporal_keys else v)
        for k, v in inputs.items()
    }
    num_steps = inputs['cells'].shape[0]
    pred_pos = _rollout(model, initial_state, num_steps)
   

    error_pos_rmse = tf.sqrt(tf.reduce_mean(tf.reduce_sum((pred_pos - inputs['world_pos'])**2, axis=-1), -1))

    scalars = {'p%d' % horizon: tf.reduce_mean(error_pos_rmse[1:horizon+1]).numpy() * 1E3 for horizon in [1, 50, error_pos_rmse.shape[0]]}

    traj_ops = {
        'cells': inputs['cells'],
        'mesh_pos': inputs['mesh_pos'],
        'gt_pos': inputs['world_pos'],
        'pred_pos': pred_pos,
        'node_type': inputs['node_type']
    }

    return scalars, traj_ops
