import setuptools
import os

VERSION = 0.2
# when changed, remember to change chencrafts/__init__.py
PACKAGES = [
    'chencrafts',
    'chencrafts.toolbox',
    'chencrafts.bsqubits',
    'chencrafts.cqed',
    'chencrafts.specfit',
    'chencrafts.projects',
]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, "requirements.txt")) as requirements:
    INSTALL_REQUIRES = requirements.read().splitlines()

setuptools.setup(name='chencrafts', 
    version=VERSION,
    description='Danyang Chen\'s personal toolbox',
    url='https://github.com/Harrinive/chencrafts',
    author='Danyang Chen',
    author_email='DanyangChen2026@u.northwestern.edu',
    license='MIT',
    packages=PACKAGES,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
)

def write_version_py(filename="chencrafts/version.py"):
    if os.path.exists(filename):
        os.remove(filename)
    versionfile = open(filename, "w")
    try:
        versionfile.write(
            f"# THIS FILE IS GENERATED FROM chencrafts SETUP.PY\n"
            "version = '{VERSION}'"
        )
    finally:
        versionfile.close()
write_version_py()