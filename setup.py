from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("auxiliary_funcs", ["auxiliary_funcs.pyx"], include_dirs=[np.get_include()]),
    Extension("collisions", ["collisions.pyx"], include_dirs=[np.get_include()]),
    Extension("monte_carlo", ["monte_carlo.pyx"], include_dirs=[np.get_include()]),
]

setup(
    name="AlphaChanneling",
    ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}),
)
