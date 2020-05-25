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
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer
from utils import Config, MathDatasetReader
from allennlp.data.iterators import BucketIterator
from sklearn.model_selection import train_test_split


config = Config(
    batch_size=32,
    max_seq_len=160,
    max_vocab_size=100000,
)

reader = MathDatasetReader(source_tokenizer=CharacterTokenizer(),
                           target_tokenizer=CharacterTokenizer(),
                           source_token_indexers={'tokens': SingleIdTokenIndexer()},
                           target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
train_dataset = reader.read('data/mathematics_dataset-v1.0/train-easy/arithmetic__add_or_sub.txt')
train_dataset, val_dataset = train_test_split(train_dataset, p=0.5)
test_dataset = reader.read('data/mathematics_dataset-v1.0/interpolate/arithmetic__add_or_sub.txt')

vocab = Vocabulary.from_instances(train_dataset, max_vocab_size=config.max_vocab_size)
iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("input_tokens", "num_tokens")])
iterator.index_with(vocab)


