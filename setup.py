from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

#get the req form requirements.txt
def get_requirements(file_path:str )->List[str]:
    '''this function will return a list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[line.replace("\n","") for line in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
    name='gradAdmissionMlProject',
    version='0.0.1',
    author='Eddie Xiao',
    author_email='eddiexiao2019@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)
