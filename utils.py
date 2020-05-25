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
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


class MathDatasetReader(Seq2SeqDatasetReader):

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
            yield self.text_to_instance(question, answer)