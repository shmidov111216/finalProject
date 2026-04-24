from setuptools import Extension, setup

module = Extension(
    "symnmfmodule",
    sources=['symnmfmodule.c', 'matrix_util.c', 'symnmf.c', 'parse_input.c'],
    include_dirs=['.'],
    define_macros=[('PYTHON_BUILD', '1')],
)

setup(
    name='symnmfmodule',
    version='1.0',
    description='Python wrapper for custom C extension',
    ext_modules=[module]
)