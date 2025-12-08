from setuptools import setup, find_packages

setup(
    name='quantlib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
        'numba>=0.57',
        'pandas>=2.0',
        'matplotlib>=3.5',
        'plotly>=5.0',
    ],
    python_requires='>=3.8',
    author='Your Name',
    description='Production-quality options pricing library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
