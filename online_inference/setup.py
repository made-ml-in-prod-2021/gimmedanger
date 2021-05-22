from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='online_inference',
    packages=find_packages(),
    version='0.1.0',
    description='HW2. MADE. ML in production',
    author='Vladimir Nazarov',
    install_requires=required,
    license='MIT',
)