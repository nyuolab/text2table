# Most of the code is taken from Huggingface implementation: https://github.com/huggingface/transformers/blob/v4.21.0/src/transformers/data/data_collator.py#L49
# In compliance with the Apache License, Version 2.0, I have modified the code to fit the needs of this project.
#
# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import torch
from collections.abc import Mapping
from torch.nn.utils.rnn import pad_sequence

# The parts of the code that are not modified are not commented. Please refer to the original code for more information.
def data_collator(features):

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                # special handling for input ids, attention mask, and global attention mask
                if k == "input_ids" or k == "attention_mask" or k == "global_attention_mask":
                    # if it is input ids, we set the padding value to the id of the pad token
                    # otherwise, we set the padding value to 0
                    if k == "input_ids":
                        padding_value = 1
                    else:
                        padding_value = 0
                    # pad the batch of groups of sequences
                    batch[k] = pad_sequence([f[k] for f in features], batch_first=True, padding_value=padding_value)
                    assert batch[k].shape[0] == len(features)
                    assert batch[k].shape[1] == max(len(f[k]) for f in features)
                    assert batch[k].shape[2] == max(len(f[k][0]) for f in features)
                else:
                    # other inputs does not need to be handled specially, so we just follow what the source code does.
                    batch[k] = torch.stack([f[k] for f in features])
            else:
                # eveything is the same here except that we are dealing with lists.
                if k == "input_ids" or k == "attention_mask" or k == "global_attention_mask":
                    if k == "input_ids":
                        padding_value = 1
                    else:
                        padding_value = 0
                    # converting lists to tensors before padding them.
                    batch[k] = pad_sequence([torch.stack(f[k]) for f in features], batch_first=True, padding_value=padding_value)
                    assert batch[k].shape[0] == len(features)
                    assert batch[k].shape[1] == max(len(f[k]) for f in features)
                    assert batch[k].shape[2] == max(len(f[k][0]) for f in features)
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

    return batch