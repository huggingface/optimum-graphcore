import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/version.py
try:
    filepath = "optimum/graphcore/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


INSTALL_REQUIRES = [
    "transformers==4.25.1",
    "optimum==1.6.1",
    "datasets",
    "tokenizers",
    "torch @ https://download.pytorch.org/whl/cpu/torch-1.13.0%2Bcpu-cp38-cp38-linux_x86_64.whl",
    "sentencepiece",
    "scipy",
    "pillow",
]

QUALITY_REQUIRES = [
    "black",
    "isort",
    "hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git",
]

EXTRA_REQUIRE = {
    "testing": [
        "filelock",
        "GitPython",
        "parameterized",
        "psutil",
        "pytest",
        "pytest-pythonpath",
        "pytest-xdist",
        "librosa",
        "soundfile",
    ],
    "quality": QUALITY_REQUIRES,
}

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
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRA_REQUIRE,
    include_package_data=True,
    zip_safe=False,
)
