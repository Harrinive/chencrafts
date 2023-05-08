from setuptools import setup, find_packages

setup(
    name='ChenCrafts',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scqubits",
        "matplotlib",
        "scipy",
        "dill",
        "tqdm",
    ],
    # entry_points={
    #     'console_scripts': [
    #         # Add any command-line scripts here that should be installed with your package.
    #         # For example:
    #         # 'my-command=my_package.cli:main',
    #     ],
    # },
    author='Danyang Chen',
    description='Danyang Chen\'s personal toolbox',
    url='https://github.com/Harrinive/chencrafts',
    author_email='DanyangChen2026@u.northwestern.edu',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # url='<url-to-your-package-repo>',
    # classifiers=[
    #     'Development Status :: 3 - Alpha',
    #     'Intended Audience :: Developers',
    #     'License :: OSI Approved :: MIT License',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: 3.6',
    #     'Programming Language :: Python :: 3.7',
    # ],
)
