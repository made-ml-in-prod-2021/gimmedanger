from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='airflow_dags.train_model',
    packages=find_packages(),
    version='0.1.0',
    description='HW3. MADE. ML in production',
    author='Vladimir Nazarov',
    install_requires=required,
    license='MIT',
)
