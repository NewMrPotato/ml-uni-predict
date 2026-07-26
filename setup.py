from setuptools import setup, find_packages


setup(
    name="unipredict",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
    extras_require={
        "torch": ["torch"],
        "tensorflow": ["tensorflow"],
    },
    author="NewMrPotato",
    description="Универсальный класс для инференса ML-моделей с гибкими форматами входных данных",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NewMrPotato/ml-uni-predict",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)