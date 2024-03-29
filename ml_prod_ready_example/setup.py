from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='ml_project',
    packages=find_packages(),
    version='0.1.0',
    description='Example of ml project',
    author='Vladimir Nazarov',
    install_requires=required,
    license='MIT',
)