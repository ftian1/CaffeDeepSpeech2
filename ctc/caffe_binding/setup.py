import os
import sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np 

try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

warp_ctc_path = "../build"
if "WARP_CTC_PATH" in os.environ:
    warp_ctc_path = os.environ["WARP_CTC_PATH"]
if not os.path.exists(os.path.join(warp_ctc_path, "libwarpctc.so")):
    print(("Could not find libwarpctc.so in {}.\n"
           "Build warp-ctc and set WARP_CTC_PATH to the location of"
           " libwarpctc.so (default is '../build')").format(warp_ctc_path))
    sys.exit(1)
ctc_include_dir = os.path.realpath('../include')

ext_modules = [
    Extension(
        'ctc',
	language='c++',
	sources=['src/binding.cpp', 'warpctc_caffe/__init__.pyx'],
        include_dirs = [numpy_include, ctc_include_dir],
	library_dirs = [os.path.realpath(warp_ctc_path)],
	libraries=['warpctc'],
	extra_link_args=['-Wl,-rpath,' + os.path.realpath(warp_ctc_path)],
        extra_compile_args=['-Wno-cpp', '-std=c++11', '-fPIC'],
    ),
]

setup(
    name='warp_caffe',
    ext_modules=ext_modules,
    cmdclass = {'build_ext': build_ext},
)
