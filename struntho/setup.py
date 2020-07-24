import sys
import os

# from sklearn._build_utils import cythonize_extensions
# from sklearn._build_utils.deprecated_modules import (
#     _create_deprecated_modules_files
# )


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    libraries = []
    if os.name == 'posix':
        libraries.append('m')

    # _create_deprecated_modules_files()

    config = Configuration('struntho', parent_package, top_path)

    # submodules with build utilities
    """ Example: 
    config.add_subpackage('__check_build')
    config.add_subpackage('_build_utils') """

    # submodules which do not have their own setup.py
    # we must manually add sub-submodules & tests
    """ Example:
    config.add_subpackage('compose')
    config.add_subpackage('compose/tests') """
    config.add_subpackage('datasets')
    config.add_subpackage('inference')
    config.add_subpackage('learners')
    config.add_subpackage('models') 
    config.add_subpackage('utils')

    # submodules which have their own setup.py
    """ Example:
    config.add_subpackage('cluster')
    config.add_subpackage('datasets') """

    # add cython extension module for isotonic regression
    """ Example:
    config.add_extension('_isotonic',
                         sources=['_isotonic.pyx'],
                         include_dirs=[numpy.get_include()],
                         libraries=libraries,
                         ) """

    # add the test directory
    """ Example:
    config.add_subpackage('tests') """

    # Skip cythonization as we do not want to include the generated
    # C/C++ files in the release tarballs as they are not necessarily
    # forward compatible with future versions of Python for instance.
    """
    if 'sdist' not in sys.argv:
        cythonize_extensions(top_path, config) """

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())