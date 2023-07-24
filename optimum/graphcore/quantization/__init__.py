from pathlib import Path
from ..custom_ops.utils import load_lib

load_lib(Path(__file__).parent.parent / "custom_ops/group_quantize_decompress/group_quantize_decompress.cpp")
# from optimum.graphcore.custom_ops.sdk_version_hash import sdk_version_hash
# print(sdk_version_hash())

from .group_quantize import group_quantize_compress, group_quantize_decompress, GroupQuantLinear
