# Optimum Graphcore

🤗 Optimum Graphcore is the interface between the 🤗 Transformers library and [Graphcore IPUs](https://www.graphcore.ai/products/ipu).
It provides a set of tools enabling model parallelization and loading on IPUs, training and fine-tuning on all the tasks already supported by Transformers while being compatible with the Hugging Face Hub and every model available on it out of the box.

## What is an Intelligence Processing Unit (IPU)?
Quote from the Hugging Face [blog post](https://huggingface.co/blog/graphcore#what-is-an-intelligence-processing-unit):
>IPUs are the processors that power Graphcore’s IPU-POD datacenter compute systems. This new type of processor is designed to support the very specific computational requirements of AI and machine learning. Characteristics such as fine-grained parallelism, low precision arithmetic, and the ability to handle sparsity have been built into our silicon.

> Instead of adopting a SIMD/SIMT architecture like GPUs, Graphcore’s IPU uses a massively parallel, MIMD architecture, with ultra-high bandwidth memory placed adjacent to the processor cores, right on the silicon die.

> This design delivers high performance and new levels of efficiency, whether running today’s most popular models, such as BERT and EfficientNet, or exploring next-generation AI applications.

> Software plays a vital role in unlocking the IPU’s capabilities. Our Poplar SDK has been co-designed with the processor since Graphcore’s inception. Today it fully integrates with standard machine learning frameworks, including PyTorch and TensorFlow, as well as orchestration and deployment tools such as Docker and Kubernetes.

> Making Poplar compatible with these widely used, third-party systems allows developers to easily port their models from their other compute platforms and start taking advantage of the IPU’s advanced AI capabilities.

## Install
To install the latest release of this package:

`pip install optimum[graphcore]`

Optimum Graphcore is a fast-moving project, and you may want to install from source:

`pip install git+https://github.com/huggingface/optimum-graphcore.git`

Last but not least, don't forget to install requirements for every example:

`cd <example-folder>
pip install -r requirements.txt`

## Supported Models
Currently the following model architectures are supported:

- BERT (base and large)
- RoBERTa (base and large)
- Vision Transformer
