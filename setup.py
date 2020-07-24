from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

include_dirs = [np.get_include()]

extensions = [
            # cython blas
    Extension("struntho.utils._cython_blas",
            ["struntho/utils/_cython_blas.pyx"]  # source files of extensiom
            ),
            # cython inference
    Extension("struntho.utils._cython_inference",
            ["struntho/utils/_cython_inference.pyx"]
            ),
            # maxmin spmp multiclass
    Extension("struntho.inference._maxmin_spmp_multiclass",
            ["struntho/inference/_maxmin_spmp_multiclass.pyx"], # source files
            include_dirs=include_dirs
            ),
            # sum product for chains
            Extension("struntho.inference._sum_product_chain",
            ["struntho/inference/_sum_product_chain.pyx"], # source files
            include_dirs=include_dirs
            ),
            # maxmin spmp multiclass
    Extension("struntho.inference._maxmin_spmp_sequence",
            ["struntho/inference/_maxmin_spmp_sequence.pyx"], # source files
            include_dirs=include_dirs
            )
]
setup(
    name="struntho",
    ext_modules=cythonize(extensions),
    # ext_modules=extensions,
    packages=['struntho', 'struntho.learners', 'struntho.inference',
                'struntho.models', 'struntho.utils', 'struntho.datasets',
                'struntho.tests', 'struntho.tests.test_learners',
                'struntho.tests.test_models', 'struntho.tests.test_inference',
                'struntho.tests.test_utils']
)