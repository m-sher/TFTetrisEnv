from setuptools import setup, find_packages

setup(
    name='TFTetrisEnv',
    version='0.1.0',
    description='TensorFlow environment for RL with Tetris SRS+',
    author='Michael Sherrick',
    author_email='michael.a.sherrick@gmail.com',
    url='https://github.com/m-sher/TFTetrisEnv',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'tf-agents',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)