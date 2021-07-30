import torch
import numpy as np
from nltk.corpus import stopwords
import string

from .config import MiscArgument, DataArguments, ModelArguments, TrainingArguments
from .model import ClassifyModel

class ClassifyDataset(torch.utils.data.Dataset):
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

class MaskedTokenLabeller():
    def __init__(self, misc_args:MiscArgument, data_args:DataArguments, model_args:ModelArguments, trainin_args:TrainingArguments) -> None:
        self._misc_args = misc_args
        self._model = None
        self._model_args = model_args
        self._data_args = data_args
        self._training_args = trainin_args
        self._load_model()

    def _load_model(self):
        if self._model_args.model_type == 'bert':
            self._model = ClassifyModel(self._model_args, self._data_args, self._training_args)


    def label_sentence(self, sentence):
        label, label_score, sentence_set = self._label_sentence(sentence)
        return label, label_score, sentence_set

    def _label_sentence(self, sentence):
        if self._model_args.model_type == 'bert':
            return self._bert_attention_label(sentence)

    def _bert_attention_label(self, sentence):
        sentence_set = set()
        result =  self._model.predict(sentence)
        tokenized_inputs = result['tokenized_inputs']['input_ids'].to("cpu").numpy()[0][1:-1]
        logits = result['logits']
        scores = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        label = scores.argmax()
        label_score = scores.max()
        attentions = result['attentions']
        for attention in attentions:
            cleaned_attentions = attention.to("cpu")[:,:,0,1:-1].squeeze().numpy()
            for cleaned_attention in cleaned_attentions:
                max_idx = cleaned_attention.argmax()
                tokens = self._model.tokenizer.convert_ids_to_tokens(tokenized_inputs)
                chosen_token = tokens[max_idx]
                if chosen_token[0] == '#' or chosen_token in stopwords.words('english') or chosen_token in string.punctuation or (max_idx < len(tokens) - 1 and tokens[max_idx + 1][0] == '#'):
                    continue
                else:
                    tokens[max_idx] = '[MASK]'
                    sentence = self._model.tokenizer.convert_tokens_to_string(tokens)
                    sentence_set.add(sentence)
        return label, label_score, sentence_set
