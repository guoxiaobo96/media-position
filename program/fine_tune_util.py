from numpy.lib.arraysetops import isin
from sklearn.metrics.pairwise import cosine_distances
from transformers.modeling_outputs import MaskedLMOutput
from scipy.sparse.construct import random
import numpy as np
import json
from typing import Optional, List, Dict, Tuple, Any, NewType, Union
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import os
import pickle
from packaging import version

import transformers
from transformers import BertTokenizer, BatchEncoding, PreTrainedTokenizer, DataCollatorForLanguageModeling
from transformers.file_utils import is_sagemaker_mp_enabled, is_apex_available

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


class MLMConsistencyDataset(torch.utils.data.Dataset):
    def __init__(self, ori_encodings, aug_encodings):
        self.ori_encodings = ori_encodings
        self.aug_encodings = aug_encodings

    def __getitem__(self, idx):
        item = dict()
        ori_item = {key: torch.tensor(val[idx])
                for key, val in self.ori_encodings.items()}
        aug_item = {key: torch.tensor(val[idx])
                for key, val in self.aug_encodings.items()}
        item['original'] = ori_item
        item['augmentation'] = aug_item
        return item

    def __len__(self):
        return int(len(self.ori_encodings['input_ids']))


class DataCollatorForLanguageModelingConsistency(DataCollatorForLanguageModeling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # original_exampls =  deepcopy(examples)
        examples_temp = list()


        if 'original' in examples[0]:
            for example in examples:
                examples_temp.append(example['original'])
            for example in examples:
                examples_temp.append(example['augmentation'])
            examples = deepcopy(examples_temp)
        else:
            examples = examples

        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {"input_ids": self._collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)}

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        temp = batch["input_ids"].numpy()
        return batch

    def _collate_batch(self, examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
        """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
        # Tensorize if necessary.
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]

        # Check if padding is necessary.
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
            return torch.stack(examples, dim=0)

        # If yes, check if we have a `pad_token`.
        if tokenizer._pad_token is None:
            raise ValueError(
                "You are attempting to pad samples but the tokenizer you are using"
                f" ({tokenizer.__class__.__name__}) does not have a pad token."
            )

        # Creating the full tensor and filling it with our data.
        max_length = max(x.size(0) for x in examples)
        if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
        result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
        for i, example in enumerate(examples):
            if tokenizer.padding_side == "right":
                result[i, : example.shape[0]] = example
            else:
                result[i, -example.shape[0] :] = example
        return result

class Trainer(transformers.Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.
        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.use_amp else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        if self.args.loss_type == 'mlm':
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss(model, inputs)
            else:
                loss = self.compute_loss(model, inputs)
        elif self.args.loss_type == 'mlm_con':
            if self.use_amp:
                with autocast():
                    loss = self.compute_loss_consistency(model, inputs)
            else:
                loss = self.compute_loss_consistency(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss_consistency(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Except for the traditional loss, we add the cosine similarity 
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        size = int(inputs['input_ids'].size()[0] / 2)
        inputs_ori = dict()
        inputs_aug = dict()

        for k, v in inputs.items():
            inputs_ori[k] = inputs[k][:size]
            inputs_aug[k] = inputs[k][size:]

        outputs_ori, sequence_output_ori = model(**inputs_ori)
        outputs_aug, sequence_output_aug = model(**inputs_aug)
        # Save past state if it exists

        if self.args.past_index >= 0:
            self._past = outputs_ori[self.args.past_index]

        if labels is not None:
            loss_ori = self.label_smoother(outputs_ori, labels)
            loss_aug = self.label_smoother(outputs_aug, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss_ori = outputs_ori["loss"] if isinstance(outputs_ori, dict) else outputs_ori[0]
            loss_aug = outputs_aug["loss"] if isinstance(outputs_aug, dict) else outputs_aug[0]

        loss = loss_aug + 0*loss_ori + cosine_loss(sequence_output_ori,sequence_output_aug)

        return (loss, outputs_ori) if return_outputs else loss

class SentenceReplacementDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_scores(scores, encodings: BatchEncoding):
    scores = [[score for score in doc] for doc in scores]
    encoded_scores = []
    input_ids = list()
    token_type_ids = list()
    attention_mask = list()
    encoding_list = list()
    error_count = 0

    for (doc_scores, doc_input_ids, doc_attemtion_mask, doc_token_type_ids, doc_offset, doc_encoding) in zip(scores, encodings.input_ids, encodings.attention_mask, encodings.token_type_ids, encodings.offset_mapping, encodings._encodings):
        # create an empty array of -100
        try:
            doc_enc_scores = np.ones(len(doc_offset), dtype=float) * -100
            arr_offset = np.array(doc_offset)

            # set labels whose first offset position is 0 and the second is not 0
            doc_enc_scores[(arr_offset[:, 0] == 0) & (
                arr_offset[:, 1] != 0)] = doc_scores
            encoded_scores.append(doc_enc_scores.tolist())
            input_ids.append(doc_input_ids)
            token_type_ids.append(doc_token_type_ids)
            attention_mask.append(doc_attemtion_mask)
            encoding_list.append(doc_encoding)
        except:
            error_count += 1

    data = {'input_ids': input_ids, 'token_type_ids': token_type_ids,
            'attention_mask': attention_mask}
    encodings = BatchEncoding(data, encoding_list)

    return encoded_scores, encodings

def cosine_loss(p, q):
    loss = 1 - F.cosine_similarity(p,q).mean()
    return loss