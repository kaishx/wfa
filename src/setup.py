from setuptools import setup, Extension
import pybind11
import sys

# to determine platform-specific compile args
if sys.platform == 'win32':
    compile_args = ['/O2', '/arch:AVX2', '/MP']
elif sys.platform in ('linux', 'darwin'):
    compile_args = ['-O3', '-march=native', '-ffast-math', '-std=c++17']
else:
    compile_args = []

cpp_module = Extension(
    'cpp_accelerator',
    sources=['backtest_core.cpp'],
    include_dirs=[
        pybind11.get_include(),
        pybind11.get_include(user=True),
    ],
    language='c++',
    extra_compile_args=compile_args,
)

setup(
    name='cpp_accelerator',
    version='1.0',
    description='C++ accelerator for WFA (Pairs Trading Core)',
    ext_modules=[cpp_module],
)


# for others - how to build the module:
# run in your environment: pip install . --no-build-isolation --force-reinstall