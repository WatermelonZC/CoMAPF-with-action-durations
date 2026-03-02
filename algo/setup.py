# setup.py (v2 - 添加 /utf-8 标志)
import platform
import sys
from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("错误: pybind11 未找到。请运行 'pip install pybind11[global]'")
    sys.exit(1)

try:
    import numpy
except ImportError:
    print("错误: numpy 未找到。请运行 'pip install numpy'")
    sys.exit(1)

if platform.system() == "Windows":
    compile_args = ["/std:c++17", "/O2", "/utf-8"]
else:
    compile_args = ["-std=c++17", "-O3", "-fvisibility=hidden"]

ext_modules = [
    Pybind11Extension(
        "planner_lib",
        ["low_level_planner.cpp"],
        extra_compile_args=compile_args
    ),
]

setup(
    name="planner_lib",
    version="0.0.1",
    author="Your Name",
    description="C++ accelerator for ECoCBS (A*, Dijkstra, Conflict)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    include_dirs=[
        numpy.get_include(),
    ],
    zip_safe=False,
    python_requires=">=3.8",
)
