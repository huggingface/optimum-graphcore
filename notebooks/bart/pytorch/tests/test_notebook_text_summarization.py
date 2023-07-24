# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import pathlib
import os
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat
import pytest

### Pytest markers
@pytest.mark.ipus("2")
def test_bart_large(tmp_path):
    """
    Test the notebook for text summarization with BART-L model (notebook_text_summarization.ipynb)
    which is meant for Paperspace.
    """
    working_path = pathlib.Path(__file__).parents[1]
    os.environ["NUM_AVAILABLE_IPU"] = "4"
    notebook_filename = working_path / "notebook_text_summarization.ipynb"
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": f"{working_path}"}})
