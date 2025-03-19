### Data Preparation

- Transformers for NLP tasks work with sentences encoded as `tokens`. The conversion is performed by `Tokenizers`.
- [Example usage of various Tokenizers](./cheat_sheets/tokenizer_cheat_sheet.pdf)
- Typical workflow with PyTorch:
  - build a tokenizer (e.g : `get_tokenizer` from torchtext lib, `BertTokenizer` from transformers lib, etc)
  - add a function that yields tokenized output for an input sentence. This function consumes an iterator.
```
def yield_tokens(sentence):
  for _, word in sentence:
    yield get_tokenizer(word)
my_iter = yield_tokens(dataset)
```
  - build vocabulary on the dataset, handling out-of-vocabulary (OOV) words using `<unk>`
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
- Special tokens can be added in by different libs. `spacy` adds `<bos>` and `<eos>` for begin and end of a sentence
- Use `<pad>` token to get similar context length for each sentence.

- Another example:
```
lines = ["LLMs taught me tokenization", 
         "Special tokenizers are ready and they will blow your mind", 
         "just saying hi!"]

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')

tokens = []
max_length = 0

for line in lines:
    tokenized_line = tokenizer_en(line)
    tokenized_line = ['<bos>'] + tokenized_line + ['<eos>']
    tokens.append(tokenized_line)
    max_length = max(max_length, len(tokenized_line))

for i in range(len(tokens)):
    tokens[i] = tokens[i] + ['<pad>'] * (max_length - len(tokens[i]))

print("Lines after adding special tokens:\n", tokens)

# Build vocabulary without unk_init
vocab = build_vocab_from_iterator(tokens, specials=['<unk>'])
vocab.set_default_index(vocab["<unk>"])

# Vocabulary and Token Ids
print("Vocabulary:", vocab.get_itos())
print("Token IDs for 'tokenization':", vocab.get_stoi())
```

#### Tokenizer types
- Word / Character / sub-word based
- sub-word based techniques rely on word frequency in a corpus to define sub-word based tokens. E.gs include :
  - `WordPiece` - doesn't select the most frequent symbol pair but rather the one that maximizes t.he likelihood of the training data when added to the vocabulary,
  - `Unigram` - starts with a large list of possibilities and gradually narrowing it down based on how frequently those pieces appear in the text
  - `SentencePiece` - handles subword segmentation and ID assignment, while Unigram's principles guide the vocabulary reduction process.

#### DataLoader
- Optimizes data batches for ML training runs by performing Batch and Transform operations.
```
# define custom dataset obj
sentences = ...
custom_dataset = CustomDataset(sentences)

# define tokenizer
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iter(map(tokenizer, sentences))

def collate_fn(batch):
  tensor_batch = []
  for sample in batch:
    # get tokens
    tokens = tokenizer(sample)
    # build tensor batch
    tensor_batch.append([vocab[token] for token in tokens])

  # make all batch elements of same length
  padded_batch = pad_sequence(tensor_batch, batch_first = True)
  return padded_batch

# Build Dataloader
dataloader = Dataloader(dataset=custom_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)

```
