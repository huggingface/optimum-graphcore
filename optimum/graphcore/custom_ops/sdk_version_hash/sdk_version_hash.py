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


import cppimport.import_hook  # noqa: F401


__all__ = ["sdk_version_hash"]


def sdk_version_hash() -> str:
    """Graphcore SDK version hash (sanitised output from C++ function `poplar::packageHash`)"""
    from . import sdk_version_hash_lib

    return sdk_version_hash_lib.sdk_version_hash()
