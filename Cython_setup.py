from setuptools import setup

from setup import build_setup_kwargs

if __name__ == "__main__":
    setup(**build_setup_kwargs(use_cython=True))
