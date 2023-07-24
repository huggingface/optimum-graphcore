# Copyright 2021 The HuggingFace Team. All rights reserved.
# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    "transformers==4.29.2",
    "optimum==1.6.1",
    "diffusers[torch]==0.12.1",
    "cppimport==22.8.2",
    "datasets",
    "tokenizers",
    "typeguard",
    "sentencepiece",
    "scipy",
    "pillow",
]

QUALITY_REQUIRES = [
    "black~=23.1",
    "isort>=5.5.4",
    "hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git",
    "ruff>=0.0.241,<=0.0.259",
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
    package_data={"": ["*.cpp", "*.hpp"]}
)
