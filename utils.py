from pathlib import Path
from typing import *
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from functools import partial
from overrides import overrides
from allennlp.data import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.nn import util as nn_util

from allennlp.common.checks import ConfigurationError

from allennlp.data.vocabulary import Vocabulary
from allennlp.data.dataset_readers import DatasetReader

from allennlp.data.fields import TextField, MetadataField, ArrayField

from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


class MathDatasetReader(DatasetReader):
    def __init__(self, config: Dict) -> None:
        super().__init__(lazy=False)
        self.config = config
        self.token_indexers = {"tokens": SingleIdTokenIndexer()}

    def tokenizer(self, x: str):
        return [c for c in CharacterTokenizer().tokenize(x)[:self.config.max_seq_len]]

    @overrides
    def text_to_instance(self, input_tokens: List[Token], output_tokens: List[Token]) -> Instance:
        input_tokens_field = TextField(input_tokens, self.token_indexers)
        fields = {"input_tokens": input_tokens_field}
        output_tokens_field = TextField(output_tokens, self.token_indexers)
        fields["output_tokens"] = output_tokens_field
        return Instance(fields)

    @overrides
    def _read(self, filepath: str, limit=1000) -> Iterator[Instance]:
        with open(filepath) as f:
            lines = f.readlines()
        if limit:
            num_pairs = min(len(lines) // 2, limit)
        else:
            num_pairs = len(lines) // 2
        for i in range(0, 2 * num_pairs, 2):
            question = lines[i].strip()
            answer = lines[i + 1].strip()
            yield self.text_to_instance(
                [x for x in self.tokenizer(question)],
                [x for x in self.tokenizer(answer)]
            )
