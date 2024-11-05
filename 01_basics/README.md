### Transformers
3 basic concepts : Tokenization -> Embedding -> Positional Encoing -> Inference -> Output Probabilities

#### Usage in LLMs
- As a `BaseModel` : Pretrained on massive datasets
- As a `Fine-Tuned Model` : fine tuned on domain and optionally also includes reinforcement learning from human feedback (e.g : GPT3+)

#### Training Procedure
1. `Tokenization` : refers to converting words / characters into numbers based on a vocabulary of say size V. e.g. : [HuggingFace example](https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling) has V of size 100069.
2. `Embedding` : convert tokens to d sized vector based on a lookup. Lookup table is of size `V x d`, each embedded representation is of size `1 x d`.
3. `Input Batch` :   
    a. A set of sentences are chosen in a batch with a defined `context length`, which is the max size of a sentence. Say we have `B x C`, where B is the batch size, C the context length.   
    b. For each sentence / element in the batch, build the embedding using the lookup. Each sentence of size `C x 1` gets converted to `C x d` after lookup.   
    c. Do this for all sentences, and we end up with `B x C x d` tensor for input sentences of size `B x C`.
4. `Positional Encoding` : describes the location or position of an entity in a sequence so that each position is assigned a unique representation. and closer elements are close. The lookup is of size `C x d`. This is added to each batch element from the batch of size `B x C x d`.
