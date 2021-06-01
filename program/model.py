import logging
import warnings
import math
import os
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from glob import glob
from typing import Any, Dict, Optional, Tuple, Union
from tokenizers import Tokenizer

from torch.utils.data import ConcatDataset
import torch
import numpy as np

from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AdapterArguments,
    AdapterConfig,
    AdapterType,
    AutoConfig,
    AutoModelWithLMHead,
    AutoModelForTokenClassification,
    BertForTokenClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForPermutationLanguageModeling,
    DataCollatorForWholeWordMask,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    LineByLineTextDataset,
    LineByLineWithRefDataset,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    pipeline,
    AutoModel,
    BertPreTrainedModel,
    BertConfig,
    BertModel,
    ModelWithHeadsAdaptersMixin
)
from transformers.configuration_utils import PretrainedConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.modeling_outputs import TokenClassifierOutput

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss




from .config import DataArguments, ModelArguments
from .data import get_dataset
from .ner_util import NERDataset


class DeepModel(ABC):
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ) -> None:
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._config: Union[PretrainedConfig, Any, None] = None
        self._model: Union[Any, None] = None
        self.tokenizer: Union[PreTrainedTokenizerBase, Any] = None

        self._model_args: ModelArguments = model_args
        self._data_args: DataArguments = data_args
        self._training_args: TrainingArguments = training_args
        self._data_collator = None
        self._trainer: Trainer = None
        self._model_path = (
            self._model_args.model_name_or_path
            if self._model_args.model_name_or_path is not None and os.path.isdir(self._model_args.model_name_or_path)
            else None
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [
                -1, 0] else logging.WARN,
        )
        self._logger.setLevel(logging.INFO)

    def _load_config(self) -> None:
        if self._model_args.config_name:
            self._config = AutoConfig.from_pretrained(
                self._model_args.config_name, cache_dir=self._model_args.cache_dir)
        elif self._model_args.model_name_or_path:
            self._config = AutoConfig.from_pretrained(
                self._model_args.model_name_or_path, cache_dir=self._model_args.cache_dir)
        else:
            self._config = CONFIG_MAPPING[self._model_args.model_type]()
            self._logger.warning(
                "You are instantiating a new config instance from scratch.")

    def _load_tokenizer(self) -> None:
        if self._model_args.tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_args.tokenizer_name, cache_dir=self._model_args.cache_dir)
        elif self._model_args.model_name_or_path:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._model_args.model_name_or_path, cache_dir=self._model_args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )

    @abstractmethod
    def _load_model(self) -> None:
        pass

    @abstractmethod
    def _prepare_model(self) -> None:
        pass


class LmAdapterModel(DeepModel):
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
            adapter_args: AdapterArguments
    ) -> None:
        super().__init__(model_args, data_args, training_args)
        self._adapter_args: AdapterArguments = adapter_args
        self._language: str = ''
        self._prepare_model()

    def _load_adapter(self) -> None:
        self._language = self._adapter_args.language
        if not self._language:
            raise ValueError(
                "--language flag must be set when training an adapter")
        # check if language adapter already exists, otherwise add it
        if self._language not in self._model.config.adapters.adapter_list(AdapterType.text_lang):
            # resolve the adapter config
            adapter_config = AdapterConfig.load(
                self._adapter_args.adapter_config,
                non_linearity=self._adapter_args.adapter_non_linearity,
                reduction_factor=self._adapter_args.adapter_reduction_factor,
            )
            # load a pre-trained from Hub if specified
            if self._adapter_args.load_adapter:
                self._model.load_adapter(
                    self._adapter_args.load_adapter,
                    AdapterType.text_lang,
                    config=adapter_config,
                    load_as=self._language,
                )
            # otherwise, add a fresh adapter
            else:
                self._model.add_adapter(
                    self._language, AdapterType.text_lang, config=adapter_config)
        # Freeze all model weights except of those of this adapter & use this adapter in every forward pass
        self._model.train_adapter([self._language])

    def _load_data_collator(self) -> None:
        if self._config.model_type == "xlnet":
            self._data_collator = DataCollatorForPermutationLanguageModeling(
                tokenizer=self.tokenizer,
                plm_probability=self._data_args.plm_probability,
                max_span_length=self._data_args.max_span_length,
            )
        else:
            if self._data_args.mlm and self._data_args.whole_word_mask:
                self._data_collator = DataCollatorForWholeWordMask(
                    tokenizer=self.tokenizer, mlm_probability=self._data_args.mlm_probability
                )
            else:
                self._data_collator = DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer, mlm=self._data_args.mlm, mlm_probability=self._data_args.mlm_probability
                )

    def _load_model(self) -> None:
        self._config.return_dict = True
        if self._model_args.model_name_or_path:
            self._model = AutoModelWithLMHead.from_pretrained(
                self._model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self._model_args.model_name_or_path),
                config=self._config,
                cache_dir=self._model_args.cache_dir,
            )
        else:
            self._logger.info("Training new model from scratch")
            self._model = AutoModelWithLMHead.from_config(self._config)
        self._model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_model(self) -> None:
        self._load_config()
        self._load_tokenizer()
        self._load_model()
        if self._adapter_args.train_adapter:
            self._load_adapter()
        self._load_data_collator()

    def train(
        self,
        train_dataset: Union[LineByLineWithRefDataset,
                             LineByLineTextDataset, TextDataset, ConcatDataset],
        eval_dataset=Union[LineByLineWithRefDataset,
                           LineByLineTextDataset, TextDataset, ConcatDataset]
    ) -> None:
        self._model.train()
        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            data_collator=self._data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            prediction_loss_only=True,
            do_save_full_model=not self._adapter_args.train_adapter,
            do_save_adapters=self._adapter_args.train_adapter,
        )
        self._trainer.train(model_path=self._model_path)
        self._trainer.save_model()
        if self._trainer.is_world_master():
            self.tokenizer.save_pretrained(self._training_args.output_dir)
        if self._training_args.do_eval:
            self._eval()

    def _eval(self) -> Dict:
        results = {}
        self._logger.info("*** Evaluate ***")

        eval_output = self._trainer.evaluate()

        perplexity = math.exp(eval_output["eval_loss"])
        result = {"perplexity": perplexity}

        output_eval_file = os.path.join(
            self._training_args.output_dir, self._adapter_args.language)
        if not os.path.exists(output_eval_file):
            os.makedirs(output_eval_file)
        output_eval_file = os.path.join(
            output_eval_file, "eval_results_lm.txt")
        if self._trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                self._logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    self._logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

        return results

    def predict(self, sentence_list) -> Dict:
        self._model.eval()
        fill_mask = pipeline(task="fill-mask", model=self._model,
                             tokenizer=self.tokenizer, device=0, top_k=10)
        result_dict = dict()
        for sentence in sentence_list:
            results = fill_mask(sentence)
            result_dict[sentence] = results
        return result_dict


class NERModel(DeepModel):
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
            adapter_args: AdapterArguments
    ) -> None:
        super().__init__(model_args, data_args, training_args)
        self._adapter_args: AdapterArguments = adapter_args
        self._language: str = ''
        self._prepare_model()

    def _load_model(self) -> None:
        self._config.return_dict = True
        self._config.num_labels = 1
        if self._model_args.model_name_or_path:
            self._model = BertForScoreLabel.from_pretrained(
                self._model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self._model_args.model_name_or_path),
                config=self._config,
                cache_dir=self._model_args.cache_dir,
            )
        else:
            self._logger.info("Training new model from scratch")
            self._model = BertForScoreLabel.from_pretrained(
                self._config)
        self._model.resize_token_embeddings(len(self.tokenizer))

    def _prepare_model(self) -> None:
        self._load_config()
        self._load_tokenizer()
        self._load_model()
        # if self._adapter_args.train_adapter:
        #     self._load_adapter()
        self._load_data_collator()

    def _load_data_collator(self) -> None:
        self._data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, label_pad_token_id=1)

    def train(
        self,
        train_dataset: NERDataset,
        eval_dataset: NERDataset
    ) -> None:
        self._model.train()

        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            # data_collator=self._data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            do_save_full_model=not self._adapter_args.train_adapter,
            do_save_adapters=self._adapter_args.train_adapter,
        )
        self._trainer.train(model_path=self._model_path)
        self._trainer.save_model()
        if self._trainer.is_world_process_zero():
            self.tokenizer.save_pretrained(self._training_args.output_dir)
        if self._training_args.do_eval:
            self._eval()

    def _eval(self) -> Dict:
        results = {}
        self._logger.info("*** Evaluate ***")

        eval_output = self._trainer.evaluate()

        loss = eval_output["eval_loss"]
        result = {"loss": loss}

        output_eval_file = os.path.join(
            self._training_args.output_dir, self._adapter_args.language)
        if not os.path.exists(output_eval_file):
            os.makedirs(output_eval_file)
        output_eval_file = os.path.join(
            output_eval_file, "eval_results.txt")
        if self._trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                self._logger.info("***** Eval results *****")
                for key in sorted(result.keys()):
                    self._logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

        results.update(result)

        return results

    def predict(self, sentence_list) -> Dict:
        self._model.eval()
        fill_mask = pipeline(task="fill-mask", model=self._model,
                             tokenizer=self.tokenizer, device=0, top_k=10)
        result_dict = dict()
        for sentence in sentence_list:
            results = fill_mask(sentence)
            result_dict[sentence] = results
        return result_dict


class BertSimpleModel(DeepModel):
    def __init__(
            self,
            model_args: ModelArguments,
            data_args: DataArguments,
            training_args: TrainingArguments,
    ) -> None:
        super().__init__(model_args, data_args, training_args)
        self._prepare_model()

    def _load_model(self):
        self._config.return_dict = True
        if self._model_args.model_name_or_path:
            self._model = AutoModel.from_pretrained(
                self._model_args.model_name_or_path,
                from_tf=bool(".ckpt" in self._model_args.model_name_or_path),
                config=self._config,
                cache_dir=self._model_args.cache_dir,
            )

    def _prepare_model(self) -> None:
        self._load_config()
        self._load_tokenizer()
        self._load_model()
        self._load_data_collator()

    def encode(self, sentence_list) -> Dict:
        self._model.eval()
        fill_mask = pipeline(task="feature-extraction",
                             model=self._model, tokenizer=self.tokenizer, device=0)
        result_dict = dict()
        for sentence in sentence_list:
            results = fill_mask(sentence)
            result_dict[sentence] = np.squeeze(results)[1:-1]
        return result_dict
        # input_ids = torch.tensor(self.tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
        # outputs = self._model(input_ids)
        # hidden_states = outputs[-1][1:]  # The last hidden-state is the first element of the output tuple
        # print("test")
        # # layer_hidden_state = hidden_states[n_layer]
        # # return layer_hidden_state


class BertForScoreLabel(ModelWithHeadsAdaptersMixin, BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # loss = None
        # if labels is not None:
        #     loss_fct = CrossEntropyLoss()
        #     # Only keep active parts of the loss
        #     if attention_mask is not None:
        #         active_loss = attention_mask.view(-1) == 1
        #         active_logits = logits.view(-1, self.num_labels)
        #         active_labels = torch.where(
        #             active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
        #         )
        #         loss = loss_fct(active_logits, active_labels)
        #     else:
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))


        loss = None
        if labels is not None:
            loss_fct = MSELoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = labels.view(-1)!=-100
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), active_logits.view(-1)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss * labels.view(-1).size()[0] / int(active_loss.sum())

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def train_adapter(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments
) -> Dict:
    model = LmAdapterModel(model_args, data_args, training_args, adapter_args)
    train_dataset = (
        get_dataset(data_args, tokenizer=model.tokenizer,
                    cache_dir=model_args.cache_dir) if training_args.do_train else None
    )
    eval_dataset = (
        get_dataset(data_args, tokenizer=model.tokenizer,
                    evaluate=True, cache_dir=model_args.cache_dir)
        if training_args.do_eval
        else None
    )
    model.train(train_dataset, eval_dataset)


def predict_adapter(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments
) -> Dict:
    model = LmAdapterModel(model_args, data_args, training_args, adapter_args)
    model.predict(["##teral"])


def encode_bert(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    adapter_args: AdapterArguments
) -> None:
    model = BertSimpleModel(model_args, data_args, training_args)
    # model.encode(["The election of Trump was bad"])
    model.encode(["teral"])


def main():
    from .config import get_config
    from .util import prepare_dirs_and_logger
    misc_args, model_args, data_args, training_args, adapter_args, analysis_args = get_config()
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, adapter_args, analysis_args)
    encode_bert(model_args, data_args, training_args, adapter_args)
    # if misc_args.task == 'train_adapter':
    #     train_adapter(model_args, data_args, training_args, adapter_args)
    # elif misc_args.task == 'predict_adapter':
    #     predict_adapter(model_args, data_args, training_args, adapter_args)


if __name__ == '__main__':
    main()
