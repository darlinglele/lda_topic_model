from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("c_lda_model", ["c_lda_model.pyx"], include_dirs=[numpy.get_include()])]

setup(
    name='c_lda_model app',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
