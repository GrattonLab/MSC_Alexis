from setuptools import setup, find_packages
#analysis/code/mL_rsfMRI/
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="MSC_Alexis",
    version="0.0.3",
    author="Alexis Porter",
    author_email="alexis.porter1313@gmail.com",
    description="Machine learning to predict states",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aporter1350/MSC_Alexis",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.5',
    install_requires=["numpy", "scipy", "sklearn", "pandas","seaborn"]
)
