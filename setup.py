from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='NanoscoPy',
    version='0.0.13',
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
        'License :: OSI Approved :: MIT License'
                ],
    # install_requires=[
    #     'numpy >= 1.20.3', 
    #     'pandas >= 1.3.1', 
    #     "SPIEPy >= 0.2.0", 
    #     "scipy >= 1.7.1",
    #     "matplotlib >= 3.4.2",
    #     "access2theMatrix >= 0.4.1",
    #     "lmfit >= 1.0.2",
    #     "pySPM >= 0.2.20"
    #     ],
    python_requires='>=3.6'
)