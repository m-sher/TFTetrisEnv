from TetrisEnv import TetrisPyEnv
import tf_agents
import numpy as np
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# tf_agents.system.multiprocessing.enable_interactive_mode()

if __name__ == '__main__':
    constructors = [lambda: TetrisPyEnv(queue_size=5, seed=123) for _ in range(4)]
    ppy_env = ParallelPyEnvironment(constructors, start_serially=False, blocking=False)
    # tf_env = TFPyEnvironment(ppy_env)
    
    timestep = ppy_env.reset()
    print(timestep)

    while not any(timestep.is_last()):
        for row in timestep.observation['board'][0]:
            print(row)
        print(timestep.observation['pieces'][0])
        action = [int(choice) for choice in input("\nMove: ").strip().split(',')]
        action = {
            'hold': np.array([action[0] for _ in range(4)]),
            'standard': np.array([action[1] for _ in range(4)]),
            'spin': np.array([action[2] for _ in range(4)])
        }
        timestep = ppy_env.step(action)
