# Text Summarization on IPUs using BART-L - Inference

The [Text Summarization on IPUs using BART-L - Inference](notebook_text_summarization.ipynb) notebook demonstrates the text summarization task with the BART-L Hugging Face Optimum model run on Graphcore IPUs.

| Framework | Domain | Model | Datasets | Tasks | Training | Inference | Reference |
|-----------|--------|-------|----------|-------|----------|-----------|-----------|
| PyTorch   | nlp | BART-L  | ?     | text summarization |  <p style="text-align: center;"> ❌ | <p style="text-align: center;">✅ <br> Min. 2 IPU (POD4) required | POD4/POD16/POD64 | link to paper/original implementation|

## Setup overview

The best way to run this demo is on Paperspace Gradient's cloud IPUs because everything is already set up for you.

To run the demo using other IPU hardware, you need to:

1. Install and enable the Poplar SDK (see [Poplar SDK setup](#poplar-sdk-setup))
2. Install the PopTorch wheel in a Python virtual environment (see [Environment setup](#environment-setup))

### Poplar SDK setup
To check if your Poplar SDK has already been enabled, run:
```bash
 echo $POPLAR_SDK_ENABLED
 ```

If no path is printed, then follow these steps to enable the Poplar SDK:
1. Navigate to your Poplar SDK root directory

2. Enable the Poplar SDK with:
```bash
cd poplar-<OS version>-<SDK version>-<hash>
. enable
```

Detailed instructions on setting up your Poplar environment are available in the [PyTorch Quick Start Guide](https://docs.graphcore.ai/projects/pytorch-quick-start/).


## Environment setup
To prepare your environment, follow these steps:

1. Create and activate a Python3 virtual environment:
```bash
python3 -m venv <venv name>
source <venv path>/bin/activate
```

2. Install the framework-specific wheels:
```bash
pip3 install ${POPLAR_SDK_ENABLED?}/../poptorch-*.whl
```
Detailed instructions on setting up your PyTorch environment are available in the [PyTorch Quick Start Guide](https://docs.graphcore.ai/projects/pytorch-quick-start).
