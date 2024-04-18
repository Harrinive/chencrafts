import os
import sys

from setuptools import setup, find_packages


VERSION = 1.1


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, "requirements.txt")) as requirements:
    INSTALL_REQUIRES = requirements.read().splitlines()
EXTRA_REQUIRES = {
    "cqed": ["torch", "multiprocess"],
    "bsqubits": ["networkx"]
}


CLASSIFIERS = """\
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: BSD License
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: MacOS
Operating System :: POSIX
Operating System :: Unix
Operating System :: Microsoft :: Windows
"""
CLASSIFIERS = [_f for _f in CLASSIFIERS.split("\n") if _f]
PLATFORMS = ["Linux", "Mac OSX", "Unix", "Windows"]


with open(os.path.join(CURRENT_DIR, 'README.md'), encoding='utf-8') as f:
    README_CONTENT = f.read().split("\n")

DESCRIPTION = README_CONTENT[0]
LONG_DESCRIPTION = "\n".join(README_CONTENT[2:])
KEYWORDS = "personal toolbox, superconducting qubits, quantum mechanics"


EXTRA_KWARGS = {}


# write a version.py file
version_path = os.path.join(CURRENT_DIR, 'chencrafts', 'version.py')
with open(version_path, "w") as versionfile:
    versionfile.write(
        f"# THIS FILE IS GENERATED FROM chencrafts SETUP.PY\n"
        f"version = '{VERSION}'"
    )


setup(name='chencrafts', 
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/Harrinive/chencrafts',
    author='Danyang Chen',
    author_email='DanyangChen2026@u.northwestern.edu',
    license='MIT',
    packages=find_packages(),
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRES,
    python_requires='>=3.10',
    classifiers=CLASSIFIERS,
    platforms=PLATFORMS,
    keywords=KEYWORDS,
    **EXTRA_KWARGS
)




