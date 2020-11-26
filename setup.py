from distutils.core import setup

setup(
    name='PySddr',
    version='0.1.dev0',
    packages=['sddr',],
    install_requires=['pandas','numpy','matplotlib','torch','statsmodels','pyyaml','patsy','scipy','Pillow','torchvision','imageio','seaborn'],
    long_description=open('README.md').read(),
)