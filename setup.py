from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "TetrisEnv.pathfinder",
            sources=["TetrisEnv/pathfinder.c"],
            extra_compile_args=["-O3", "-std=c99"],
        ),
        Extension(
            "TetrisEnv.hole_finder",
            sources=["TetrisEnv/hole_finder.c"],
            extra_compile_args=["-O3", "-std=c99"],
        ),
        Extension(
            "TetrisEnv.b2b_search",
            sources=["TetrisEnv/b2b_search.c"],
            extra_compile_args=["-O3", "-std=c99"],
            extra_link_args=["-lm"],
        ),
    ],
)
