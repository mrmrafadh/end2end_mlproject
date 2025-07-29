from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path)-> List[str]:
    """
    This function reads a requirements file and returns a list of requirements.
    """
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        return [req.replace("/n","") for req in requirements if req.strip() and not req.startswith('-')]

setup(
    name='ml_project',
    version='0.1.0',
    author='rafadh',
    author_email='mrmrafadh@gmail.com',
    description='END 2 END ML project',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)