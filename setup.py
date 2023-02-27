from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['habitat_ros'],
    package_dir={'': 'script'}
)

setup(**setup_args)