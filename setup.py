from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]


if use_cython:
    import numpy
    cy_kw = dict(
        include_dirs=[numpy.get_include(), ],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-mtune=native'], 
        extra_link_args=['-fopenmp'],
        language_level="3str"
    )
    ext_modules += [
        Extension("cska.cy.cy_kmers", [ "cska/cython/kmers.pyx" ], **cy_kw),
        Extension("cska.cy.cy_model", [ "cska/cython/model.pyx" ], **cy_kw ),
        Extension("cska.cy.cy_fastrand", [ "cska/cython/fastrand.pyx" ], **cy_kw ),
        #Extension("cska.cy_cmpxchg", [ "cska/cython/test_cmpxchg.pyx" ], extra_compile_args=['-fopenmp'], extra_link_args=['-fopenmp'], ),
    ]
    cmdclass.update({ 'build_ext': build_ext })
else:
    ext_modules += [
        Extension("cska.cy.kmers", [ "cska/cython/kmers.c" ]),
        Extension("cska.cy.model", [ "cska/cython/model.c" ]),
        Extension("cska.cy.fastrand", [ "cska/cython/fastrand.c" ]),
    ]


setup(
    name = "cska",
    version = "0.9.10",
    description='A fast Cython implementation of the "Streaming K-mer Assignment" algorithm initially described in Lambert et al. 2014 (PMID: 24837674)',
    url = 'https://bitbucket.org/marjens/cska/',
    author = 'Marvin Jens',
    author_email = 'mjens@mit.edu',
    license = 'MIT',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    keywords = 'rna RBNS k-mer kmer statistics biology bioinformatics',

    install_requires=['cython', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'zmq', 'jinja2', 'future_fstrings'],
    scripts=['bin/cska'],
    package_dir='',
    packages=['cska'],
    cmdclass = cmdclass,
    ext_modules=ext_modules,
)
