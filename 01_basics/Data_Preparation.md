#### Data Preparation

- Transformers for NLP tasks work with sentences encoded as `tokens`. The conversion is performed by `Tokenizers`.
- Typical workflow :
  - build a tokenizer (e.g : `get_tokenizer` from torchtext lib, `BertTokenizer` from transformers lib, etc)
  - add a function that yields tokenized output for an input sentence. This function consumes an iterator.
```
def yield_tokens(sentence):
  for _, word in sentence:
    yield get_tokenizer(word)
my_iter = yield_tokens(dataset)
```
  - build vocabulary on the dataset
```
vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
vocab.get_stoi()
```
  - perform tokenization
```
def get_tokens_and_indices(iterator):
  tokenized_sentence = next(iterator)
  token_indices = [vocab[token] for token in tokenized_sentence]
  return tokenized_sentence, token_indices

tokenized_sentence, token_indices = get_tokens_and_indices(my_iter)
next(my_iter) # references to tokenized_sentence, token_indices would now be updated.
```
