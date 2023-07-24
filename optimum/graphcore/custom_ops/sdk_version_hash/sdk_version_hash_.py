# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import cppimport.import_hook
import warnings
import logging

__all__ = ["sdk_version_hash"]


def sdk_version_hash() -> str:
    """Graphcore SDK version hash (sanitised output from C++ function `poplar::packageHash`)"""
    try:
        from . import sdk_version_hash_lib
        return sdk_version_hash_lib.sdk_version_hash()
    except SystemExit as error:
        warn_str = f"Failed to collect Poplar SDK package hash. Failed with error: {error}"
        logging.warning(warn_str)
        warnings.warn(warn_str)
        return "unknown"
