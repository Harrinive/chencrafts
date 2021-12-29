import setuptools

PACKAGES = [
    'chencrafts',
    'chencrafts/core'
]

setuptools.setup(name='chencrafts', 
      version='0.1',
      description='Danyang Chen\'s personal toolbox',
      url='https://github.com/Harrinive/chencrafts',
      author='Danyang Chen',
      author_email='DanyangChen2026@u.northwestern.edu',
      license='MIT',
      packages=PACKAGES,
      zip_safe=False)