from setuptools import setup, find_packages

setup(
    name='quantlib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'pandas>=2.0',
        'matplotlib>=3.5',
    ],
    python_requires='>=3.10',
    author='Alexios Philalithis',
    description='Options pricing library: analytical, tree, PDE and Monte Carlo engines',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
