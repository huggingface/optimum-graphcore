# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)

from .modeling_groupbert import (
    GroupBertConfig,
    GroupBertForMaskedLM,
    GroupBertForMultipleChoice,
    GroupBertForPreTraining,
    GroupBertForQuestionAnswering,
    GroupBertForSequenceClassification,
    GroupBertForTokenClassification,
    GroupBertModel,
)


AutoConfig.register("groupbert", GroupBertConfig)
AutoModel.register(GroupBertConfig, GroupBertModel)
AutoModelForPreTraining.register(GroupBertConfig, GroupBertForPreTraining)
AutoModelForMaskedLM.register(GroupBertConfig, GroupBertForMaskedLM)
AutoModelForMultipleChoice.register(GroupBertConfig, GroupBertForMultipleChoice)
AutoModelForQuestionAnswering.register(GroupBertConfig, GroupBertForQuestionAnswering)
AutoModelForTokenClassification.register(GroupBertConfig, GroupBertForTokenClassification)
AutoModelForSequenceClassification.register(GroupBertConfig, GroupBertForSequenceClassification)
