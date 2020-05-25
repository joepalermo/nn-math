import logging
from typing import *
import torch
from allennlp.data import Instance
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper, StackedSelfAttentionEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from overrides import overrides
logger = logging.getLogger(__name__)


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


class MathDatasetReader(Seq2SeqDatasetReader):

    def __init__(self,
                 source_tokenizer: Tokenizer = None,
                 target_tokenizer: Tokenizer = None,
                 source_token_indexers: Dict[str, TokenIndexer] = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: Optional[int] = None,
                 target_max_tokens: Optional[int] = None,
                 source_add_start_token: bool = True,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._source_tokenizer = source_tokenizer or CharacterTokenizer()
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = source_token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_add_start_token = source_add_start_token
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

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
        if self._source_max_tokens and self._source_max_exceeded:
            logger.info("In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
                        self._source_max_exceeded, self._source_max_tokens)
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info("In %d instances, the target token length exceeded the max limit (%d) and were truncated.",
                        self._target_max_exceeded, self._target_max_tokens)


def create_model(config, vocab):
    embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'), embedding_dim=config.embedding_dim)
    if config.encoder_type == 'LSTM':
        encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(config.embedding_dim, config.hidden_dim, batch_first=True))
    elif config.encoder_type == 'transformer':
        encoder = StackedSelfAttentionEncoder(input_dim=config.embedding_dim, hidden_dim=config.hidden_dim,
                                              projection_dim=128,
                                              feedforward_hidden_dim=128, num_layers=1, num_attention_heads=8)
    source_embedder = BasicTextFieldEmbedder({"tokens": embedding})
    max_decoding_steps = config.target_max_tokens
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=config.embedding_dim,
                          target_namespace='target_tokens',
                          attention=DotProductAttention(),
                          beam_size=8,
                          use_bleu=True)
    return model