from TetrisEnvs.TFTetrisEnv.TFTetrisEnv import TFTetrisEnv
import tensorflow as tf
from tf_agents.environments.batched_py_environment import BatchedPyEnvironment
from tf_agents.environments.tf_py_environment import TFPyEnvironment


class TFTetrisRunner:
    def __init__(
        self,
        queue_size,
        max_holes,
        max_len,
        num_steps,
        num_envs,
        p_model,
        v_model,
        seed=[0, 0],
    ):
        self._queue_size = queue_size
        self._max_len = max_len
        self._num_steps = num_steps
        self._num_envs = num_envs

        self.p_model = p_model
        self.v_model = v_model

        self._seed = seed

        def make_env():
            env = TFTetrisEnv(
                queue_size=queue_size, max_holes=max_holes, max_len=max_len, seed=seed
            )
            return env

        self.env = TFPyEnvironment(
            BatchedPyEnvironment([make_env for _ in range(num_envs)])
        )

    def _single_step(
        self,
        t,
        board,
        pieces,
        all_boards,
        all_pieces,
        all_actions,
        all_log_probs,
        all_values,
        all_attacks,
        all_height_penalty,
        all_hole_penalty,
        all_skyline_penalty,
        all_bumpy_penalty,
        all_death_penalty,
        all_dones,
    ):
        key_sequence, log_probs = self.p_model.predict((board, pieces))
        values = tf.squeeze(self.v_model.predict((board, pieces)), axis=-1)

        # Perform the action
        time_step = self.env.step(key_sequence)
        new_board = time_step.observation["board"]
        new_pieces = time_step.observation["pieces"]

        reward = time_step.reward
        attack = reward["attack"]
        height_penalty = reward["height_penalty"]
        hole_penalty = reward["hole_penalty"]
        skyline_penalty = reward["skyline_penalty"]
        bumpy_penalty = reward["bumpy_penalty"]
        death_penalty = reward["death_penalty"]

        dones = tf.cast(time_step.is_last(), tf.float32)

        # Store the data
        all_boards = all_boards.write(t, board)
        all_pieces = all_pieces.write(t, pieces)
        all_actions = all_actions.write(t, key_sequence)
        all_log_probs = all_log_probs.write(t, log_probs)
        all_values = all_values.write(t, values)

        # Store the penalties and rewards
        all_attacks = all_attacks.write(t, attack)
        all_height_penalty = all_height_penalty.write(t, height_penalty)
        all_hole_penalty = all_hole_penalty.write(t, hole_penalty)
        all_skyline_penalty = all_skyline_penalty.write(t, skyline_penalty)
        all_bumpy_penalty = all_bumpy_penalty.write(t, bumpy_penalty)
        all_death_penalty = all_death_penalty.write(t, death_penalty)

        all_dones = all_dones.write(t, dones)

        return (
            t + 1,
            new_board,
            new_pieces,
            all_boards,
            all_pieces,
            all_actions,
            all_log_probs,
            all_values,
            all_attacks,
            all_height_penalty,
            all_hole_penalty,
            all_skyline_penalty,
            all_bumpy_penalty,
            all_death_penalty,
            all_dones,
        )

    def collect_trajectory(self):
        all_boards = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, 24, 10, 1),
        )
        all_pieces = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, self._queue_size + 2),
        )
        all_actions = tf.TensorArray(
            dtype=tf.int64,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs, self._max_len),
        )
        all_log_probs = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_values = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_attacks = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_height_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_hole_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_skyline_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_bumpy_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_death_penalty = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )
        all_dones = tf.TensorArray(
            dtype=tf.float32,
            size=self._num_steps,
            dynamic_size=False,
            element_shape=(self._num_envs,),
        )

        time_step = self.env.reset()
        board = time_step.observation["board"]
        pieces = time_step.observation["pieces"]

        (
            t,
            board,
            pieces,
            all_boards,
            all_pieces,
            all_actions,
            all_log_probs,
            all_values,
            all_attacks,
            all_height_penalty,
            all_hole_penalty,
            all_skyline_penalty,
            all_bumpy_penalty,
            all_death_penalty,
            all_dones,
        ) = tf.while_loop(
            lambda t, nb, np, ab, ap, aa, alp, av, aat, ahp, ahop, asp, abp, adp, ad: t
            < self._num_steps,
            self._single_step,
            [
                tf.constant(0, dtype=tf.int32),
                board,
                pieces,
                all_boards,
                all_pieces,
                all_actions,
                all_log_probs,
                all_values,
                all_attacks,
                all_height_penalty,
                all_hole_penalty,
                all_skyline_penalty,
                all_bumpy_penalty,
                all_death_penalty,
                all_dones,
            ],
            parallel_iterations=1,
        )

        all_boards = all_boards.stack()
        all_pieces = all_pieces.stack()
        all_actions = all_actions.stack()
        all_log_probs = all_log_probs.stack()
        all_values = all_values.stack()
        all_attacks = all_attacks.stack()
        all_height_penalty = all_height_penalty.stack()
        all_hole_penalty = all_hole_penalty.stack()
        all_skyline_penalty = all_skyline_penalty.stack()
        all_bumpy_penalty = all_bumpy_penalty.stack()
        all_death_penalty = all_death_penalty.stack()
        all_dones = all_dones.stack()

        return (
            all_boards,
            all_pieces,
            all_actions,
            all_log_probs,
            all_values,
            all_attacks,
            all_height_penalty,
            all_hole_penalty,
            all_skyline_penalty,
            all_bumpy_penalty,
            all_death_penalty,
            all_dones,
        )
