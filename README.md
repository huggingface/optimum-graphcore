<p align="center">
    <img src="readme_logo.png" />
</p>

# Optimum Graphcore

🤗 Optimum Graphcore is the interface between the 🤗 Transformers library and [Graphcore IPUs](https://www.graphcore.ai/products/ipu).
It provides a set of tools enabling model parallelization and loading on IPUs, training and fine-tuning on all the tasks already supported by Transformers while being compatible with the Hugging Face Hub and every model available on it out of the box.

## What is an Intelligence Processing Unit (IPU)?
Quote from the Hugging Face [blog post](https://huggingface.co/blog/graphcore#what-is-an-intelligence-processing-unit):
>IPUs are the processors that power Graphcore’s IPU-POD datacenter compute systems. This new type of processor is designed to support the very specific computational requirements of AI and machine learning. Characteristics such as fine-grained parallelism, low precision arithmetic, and the ability to handle sparsity have been built into our silicon.

> Instead of adopting a SIMD/SIMT architecture like GPUs, Graphcore’s IPU uses a massively parallel, MIMD architecture, with ultra-high bandwidth memory placed adjacent to the processor cores, right on the silicon die.

> This design delivers high performance and new levels of efficiency, whether running today’s most popular models, such as BERT and EfficientNet, or exploring next-generation AI applications.

## Poplar SDK setup
A Poplar SDK environment needs to be enabled to use this library. Please refer to Graphcore's [Getting Started](https://docs.graphcore.ai/en/latest/getting-started.html) guide.

## Install
To install the latest release of this package:

`pip install optimum[graphcore]`

Optimum Graphcore is a fast-moving project, and you may want to install from source.

`pip install git+https://github.com/huggingface/optimum-graphcore.git`


## Running the examples

There are a number of examples provided in the `examples` directory. Each of these contains a README with command lines for running them on IPUs with Optimum Graphcore.

Please install the requirements for every example:

```
cd <example-folder>
pip install -r requirements.txt
```

## How to use it?
🤗 Optimum Graphcore was designed with one goal in mind: make training and evaluation straightforward for any 🤗 Transformers user while leveraging the complete power of IPUs.
There are two main classes one needs to know:
- IPUTrainer: the trainer class that takes care of compiling the model to run on IPUs, and of performing training and evaluation.
- IPUConfig: the class that specifies attributes and configuration parameters to compile and put the model on the device.

The `IPUTrainer` is very similar to the [🤗 Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer), and adapting a script using the Trainer to make it work with IPUs will mostly consists of simply swapping the `Trainer` class for the `IPUTrainer` one. That's how most of the [example scripts](https://github.com/huggingface/optimum-graphcore/tree/main/examples) were adapted from their [original counterparts](https://github.com/huggingface/transformers/tree/master/examples/pytorch).

Original script:
```python
from transformers import Trainer, TrainingArguments

# A lot of code here

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,  # Original training arguments.
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```


Transformed version that can run on IPUs:
```python
from optimum.graphcore import IPUConfig, IPUTrainer, IPUTrainingArguments

# A lot of the same code as the original script here

# Loading the IPUConfig needed by the IPUTrainer to compile and train the model on IPUs
ipu_config = IPUConfig.from_pretrained(
    training_args.ipu_config_name if training_args.ipu_config_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)

# Initialize our Trainer
trainer = IPUTrainer(
    model=model,
    ipu_config=ipu_config,
    # The training arguments differ a bit from the original ones, that is why we use IPUTrainingArguments
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
```

## Supported Models
The following model architectures and tasks are currently supported by 🤗 Optimum Graphcore:
|         | Pre-Training       | Masked LM          | Causal LM          | Seq2Seq LM (Summarization, Translation, etc) | Sequence Classification | Token Classification | Question Answering | Multiple Choice    | Image Classification |
|---------|--------------------|--------------------|--------------------|----------------------------------------------|-------------------------|----------------------|--------------------|--------------------|----------------------|
| BART    | :heavy_check_mark: |                    | ✗                  | :heavy_check_mark:                           | :heavy_check_mark:                       |                      | ✗                  |                    |                      |
| BERT    | :heavy_check_mark: | :heavy_check_mark: | ✗                  |                                              | :heavy_check_mark:      | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark: |                      |
| ConvNeXt| :heavy_check_mark: |                    |                    |                                              |                         |                      |                    |                    | :heavy_check_mark:   |
| DeBERTa | ✗                  | ✗                  |                    |                                              | :heavy_check_mark:      | :heavy_check_mark:   | :heavy_check_mark: |                    |                      |
| GPT-2   | :heavy_check_mark: |                    | :heavy_check_mark: |                                              | :heavy_check_mark:      | :heavy_check_mark:   |                    |                    |                      |
| HuBERT  | ✗                  |                    |                    |                                              | :heavy_check_mark:      |                      |                    |                    |                      |
| LXMERT  | ✗                  |                    |                    |                                              |                         |                      | :heavy_check_mark: |                    |                      |
| RoBERTa | :heavy_check_mark: | :heavy_check_mark: | ✗                  |                                              | :heavy_check_mark:      | :heavy_check_mark:   | :heavy_check_mark: | :heavy_check_mark: |                      |
| T5      | :heavy_check_mark: |                    |                    | :heavy_check_mark:                           |                         |                      |                    |                    |                      |
| ViT     | ✗                  |                    |                    |                                              |                         |                      |                    |                    | :heavy_check_mark:   |
| Wav2Vec2| :heavy_check_mark: |                    |                    |                                              |                         |                      |                    |                    |                      |

If you find any issue while using those, please open an issue or a pull request.
