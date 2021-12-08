from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='NanoscoPy',
    version='0.0.7',
    author = "Jesse Thompson, Darian Smalley",
    author_email = "darian.smalley@gmail.com",
    description = "microscopy data processor",
    long_description = readme(),
    url = 'https://github.com/darianSmalley/NanoscoPy',
    packages= find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Microscopy'
                ],
    install_requires=[
        'numpy', 
        'pandas', 
        'sklearn', 
        "SPIEPy", 
        "scipy",
        "matplotlib",
        "access2theMatrix",
        "lmfit",
        "pySPM",
        "SPM",
        "wheel"
        ],
    python_requires='>=3.6'
)