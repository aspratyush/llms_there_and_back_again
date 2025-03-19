### Overview
- one-hot encoding and bag of words
- embeddings and embeddings bags

#### One-hot encoding
- Token index to one-hot encoding

#### Bag of words
- combine one-hot encoded word into a one-hot encoded sentence.

#### Embedding layer
- replaces Linear layers that would accept a one-hot encoded input
- accepts the token index and produces an embedding representation similar to what Linear layers would have with the one-hot encoded input.

#### Embedding Matrix
- embedding weights are combined, 1 row per word, into a matrix
![Embedding Matrix Example](./imgs/embedding_matrix.png)
