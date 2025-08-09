from setuptools import setup, find_packages
# from Cython.Build import cythonize

# ext_modules = [
#     Extension("TetrisEnvs.CTetrisEnv.CMoves",
#               ["TetrisEnvs/CTetrisEnv/CMoves.pyx"],
#               include_dirs=[np.get_include()]),
#     Extension("TetrisEnvs.CTetrisEnv.CPieces",
#               ["TetrisEnvs/CTetrisEnv/CPieces.pyx"],
#               include_dirs=[np.get_include()]),
#     Extension("TetrisEnvs.CTetrisEnv.CRotationSystem",
#               ["TetrisEnvs/CTetrisEnv/CRotationSystem.pyx"],
#               include_dirs=[np.get_include()]),
#     Extension("TetrisEnvs.CTetrisEnv.CScorer",
#               ["TetrisEnvs/CTetrisEnv/CScorer.pyx"],
#               include_dirs=[np.get_include()]),
#     Extension("TetrisEnvs.CTetrisEnv.CTetrisEnv",
#               ["TetrisEnvs/CTetrisEnv/CTetrisEnv.pyx"],
#               include_dirs=[np.get_include()]),
# ]

setup(
    name="TetrisEnvs",
    version="0.4.0",
    description="Environments for RL with Tetris SRS+",
    author="Michael Sherrick",
    author_email="michael.a.sherrick@gmail.com",
    url="https://github.com/m-sher/TFTetrisEnv",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "tf-agents",
        "numpy",
    ],
    # ext_modules=cythonize(
    #     ext_modules,
    #     compiler_directives={'language_level': "3"}
    # ),
    # include_dirs=[np.get_include()],
    # zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
