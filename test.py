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
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.token_indexers import SingleIdTokenIndexer
from utils import Config, MathDatasetReader
from allennlp.data.iterators import BucketIterator


config = Config(
    max_seq_len=160,
    max_vocab_size=100000,
)

reader = MathDatasetReader(config)
train_dataset = reader.read('data/mathematics_dataset-v1.0/train-easy/arithmetic__add_or_sub.txt')
# TODO: split out val instead of using test set
validation_dataset = reader.read('data/mathematics_dataset-v1.0/interpolate/arithmetic__add_or_sub.txt')
vocab = Vocabulary.from_instances(train_dataset, max_vocab_size=config.max_vocab_size)
iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("input_tokens", "num_tokens")])
iterator.index_with(vocab)
