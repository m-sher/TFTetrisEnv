from setuptools import setup, find_packages

setup(
    name='TensorFlow-Tetris-SRS-Plus',
    version='0.1.0',
    description='TensorFlow environment for RL with Tetris SRS+',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/TensorFlow-Tetris-SRS-Plus',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'tf-agents',
        'numpy',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)