# Tokenizer
⚠️ **Important:** This is not a a "competitive" tokenizer (a tokenizer similar to the one used by the state-of-the-arte models) 
it's a basic tokenizer build it to show the basic structure of a tokenizer

Subworkd Tokenizer based on the BPE (Byte-Level paire encoding) algorithm used in LLM tokenization, check this site 
https://huggingface.co/learn/nlp-course/en/chapter6/5 if you want to know more about BPE tokenization and in general
about tokenization

## Files
### src
- tn.py: Code of the tokenizer
- tn_fn.py: External functions used by the tokenizer, there are two functions `get_pairs()` and `merge()` that are the two functions that make works the BPE algorithm

### test
- compare.py: Used during the developing process to make sure that my tokenizer was working as well as the hugging face tokenizer (that it's a famous tokenizer library)

### tokenizer save
folder where the tokenizer data are saved after calling the function `tn.save()`, there is also the function `tn.load()` in case you want load a saved tokenizer
  
## How to use it
### train tokenizer
```
from tn import Tokenizer

# import ds
ds = load_dataset('wikitext', name='wikitext-2-raw-v1', split='train')

corpus = ' '.join(ds['text'])

# train ids
tn = Tokenizer()

tn.train(corpus=corpus[:1000000], vocab_size=7000)
