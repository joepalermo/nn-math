import itertools
import torch
import torch.optim as optim
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.predictors import Seq2SeqPredictor
from allennlp.training.trainer import Trainer
from sklearn.model_selection import train_test_split
from utils import Config, MathDatasetReader, create_model

config = Config(
    source_max_tokens=160,
    target_max_tokens=30,
    max_vocab_size=1000,
    encoder_type='transformer',
    embedding_dim=32,
    hidden_dim=32,
    n_epochs=50,
    batch_size=32,
)

# prep data
reader = MathDatasetReader(source_tokenizer=CharacterTokenizer(),
                           target_tokenizer=CharacterTokenizer(),
                           source_token_indexers={'tokens': SingleIdTokenIndexer()},
                           target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')},
                           source_max_tokens=config.source_max_tokens,
                           target_max_tokens=config.target_max_tokens
                           )
train_dataset = reader.read('data/mathematics_dataset-v1.0/train-easy/arithmetic__add_or_sub.txt')
train_dataset, validation_dataset = train_test_split(train_dataset, test_size=0.5)
test_dataset = reader.read('data/mathematics_dataset-v1.0/interpolate/arithmetic__add_or_sub.txt')
vocab = Vocabulary.from_instances(train_dataset, max_vocab_size=config.max_vocab_size)

# create model
model = create_model(config, vocab)

# configure training loop
optimizer = optim.Adam(model.parameters())
iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("source_tokens", "num_tokens")])
iterator.index_with(vocab)
if torch.cuda.is_available():
    cuda_device = 0
    model = model.cuda(cuda_device)
else:
    cuda_device = -1
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_dataset,
                  validation_dataset=validation_dataset,
                  num_epochs=1,
                  cuda_device=cuda_device)

# training loop
for i in range(config.n_epochs):
    print('Epoch: {}'.format(i))
    trainer.train()
    predictor = Seq2SeqPredictor(model, reader)
    for instance in itertools.islice(validation_dataset, 10):
        print('input:', instance.fields['source_tokens'].tokens)
        print('target:', instance.fields['target_tokens'].tokens)
        print('predicted:', predictor.predict_instance(instance)['predicted_tokens'])
