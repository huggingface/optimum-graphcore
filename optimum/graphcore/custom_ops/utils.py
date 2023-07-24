# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# Source adapted from: https://github.com/graphcore/examples-utils/blob/v3.4/examples_utils/load_lib_utils/load_lib_utils.py

import ctypes
import hashlib
import os
from unittest.mock import patch

import cppimport
from cppimport.importer import (
    build_safely,
    is_build_needed,
    setup_module_data,
)

from .sdk_version_hash import sdk_version_hash


__all__ = ["load_lib"]

settings = {"file_exts": (".cpp",)}


def _calc_cur_checksum_with_sdk_version():
    version = sdk_version_hash()

    def func(file_lst, module_data):
        text = b""
        for filepath in file_lst:
            with open(filepath, "rb") as f:
                text += f.read()
        cpphash = hashlib.md5(text).hexdigest()
        hash = f"SDK-VERSION-{version}-{cpphash}"
        return hash

    return func


def _build(filepath, timeout: int = 5 * 60):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")
    filepath = os.path.abspath(filepath)

    old_timeout = cppimport.settings.get("lock_timeout", 5 * 60)
    try:
        cppimport.settings["lock_timeout"] = timeout
        # TODO: remove hack once ticket resolved: https://github.com/tbenthompson/cppimport/issues/76
        # TODO: A hack to include the SDK version hash as part of the cppimport hash
        with patch("cppimport.checksum._calc_cur_checksum", new=_calc_cur_checksum_with_sdk_version()):
            fullname = os.path.splitext(os.path.basename(filepath))[0]
            module_data = setup_module_data(fullname, filepath)
            if is_build_needed(module_data):
                build_safely(filepath, module_data)
            binary_path = module_data["ext_path"]
    finally:
        cppimport.settings["lock_timeout"] = old_timeout

    return binary_path


def load_lib(filepath: str, timeout: int = 5 * 60):
    """Compile a C++ source file using `cppimport`, load the shared library into the process using `ctypes` and
    return it.

    Compilation is not triggered if an existing binary matches the source file hash and Graphcore SDK version which is
    embedded in the binary file.

    `cppimport` is used to compile the source which uses a special comment in the C++ file that includes the
    compilation parameters. Here is an example of such a comment which defines compiler flags, additional sources files
    and library options (see `cppimport` documentation for more info):

    ```
    /*
    <%
    cfg['sources'] = ['another_source.cpp']
    cfg['extra_compile_args'] = ['-std=c++14', '-fPIC', '-O2', '-DONNX_NAMESPACE=onnx', '-Wall']
    cfg['libraries'] = ['popart', 'poplar', 'popops', 'poputil', 'popnn']
    setup_pybind11(cfg)
    %>
    */
    ```

    Its also recommended to include the cppimport header at the top of the source file `\\ cppimport` to indicate that
    it will be loaded via cppimport and so the `load_lib_all` function will build it.

    Parameters:
        filepath (str): File path of the C++ source file
        timeout (int): Timeout time if cannot obtain lock to compile the source

    Returns:
        lib: library instance. Output of `ctypes.cdll.LoadLibrary`
    """

    binary_path = _build(filepath, timeout)
    lib = ctypes.cdll.LoadLibrary(binary_path)

    return lib
