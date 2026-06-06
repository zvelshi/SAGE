from setuptools import setup, find_packages

setup(
    name="baja_suspension",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy", 
        "matplotlib",
        "PyYAML",
        "pymoo",
        "openpyxl",
        "pandas",
    ],
)
