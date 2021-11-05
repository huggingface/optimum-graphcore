import re

from setuptools import find_namespace_packages, setup

# Ensure we match the version set in optimum/version.py
try:
    filepath = "optimum/graphcore/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


install_requires = [
    "optimum",
    "datasets>=1.7.0",
    "tokenizers>=0.10.3",
    "scipy>=1.5.4",
    "pyyaml>=5.4.1",
    "wandb==0.12.1",
    "pytest",
    "pytest-pythonpath",
    "tfrecord>=1.13",
    "filelock>=3.0.12",
    # "mpi4py>=3.0.3",
    # "horovod>=0.22.0",
]

setup(
    name="optimum-graphcore",
    version=__version__,
    description="Optimum Library is an extension of the Hugging Face Transformers library, providing a framework to "
    "integrate third-party libraries from Hardware Partners and interface with their specific "
    "functionality.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, quantization, pruning, training, ipu",
    url="https://huggingface.co/hardware",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum.*"]),
    install_requires=install_requires,
    include_package_data=True,
    zip_safe=False,
)
