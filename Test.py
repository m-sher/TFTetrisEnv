from TetrisEnv import TetrisPyEnv
from tf_agents.environments import tf_py_environment

test_env = TetrisPyEnv(queue_size=5, seed=123)

tf_env = tf_py_environment.TFPyEnvironment(test_env)