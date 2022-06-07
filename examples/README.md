# Examples

The following examples are IPU ready versions of the [original transformers examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch).


## Table of Tasks

Here is the list of the examples:

| **Task**                                     | **Example datasets** |
|----------------------------------------------|----------------------|
| [language-modeling](language-modeling)       | WikiText-2           |
| [multiple-choice](multiple-choice)           | SWAG                 |
| [question-answering](question-answering)     | SQuAD                |
| [summarization](summarization)               | XSUM                 |
| [text-classification](text-classification)   | GLUE                 |
| [translation](translation)                   | WMT                  |
| [audio-classification](audio-classification) | SUPERB KS            |
| [image-classification](image-classification) | CIFAR-10             |
| [speech-pretraining](speech-pretraining)     | LibriSpeech ASR      |


## Tips

### Requirements
For each example, you will need to install the requirements before being able to run it:

```bash
cd <example-folder>
pip install -r requirements.txt
```

### Finding the right IPUConfig

Compared to transformers, one extra argument that you will need to pass to all of the examples is `--ipu_config_name`, which specifies compilation and parallelization information for a given model.
You can find an example for all the model architectures we support on the 🤗 Hub under the [Graphcore organization](https://huggingface.co/Graphcore). For instance, for `bert-base-uncased` you can use `Graphcore/bert-base-uncased`.


### Run on Spell

You can check the instructions on how to run a given example on Spell, a service that allows you to run commands using IPUs easily, [here](https://github.com/huggingface/optimum-graphcore#run-on-spell).
