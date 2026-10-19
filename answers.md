# CS336 Assignment 1 — Answers and experiment notes

Handout: Spring 2026, version 26.0.3. See `cs336_assignment1_basics.pdf` for full prompts and deliverables.

Hardware: MacBook Pro, M3 Pro, 18 GB unified memory.

### unicode1 — Understanding Unicode

#### (a)

**My answers:**

chr(0) returns the null character.

#### (b)

**My answers:**

chr(0).__repr__() returns `'\x00'`, and print(chr(0)) prints the invisible null character followed by a new line.

#### (c)

**My answers:**

The representation of chr(0) is the string of its hexadecimal representation `\x00`. When chr(0) the null character is embedded in a string and printed out, the printed outcome shows nothing visible at the place where chr(0) the null character appears. However, it does take up one character in length. Also, ord(chr(0)) is the unicode code point of chr(0) the null character, and is the integer 0.

### unicode2 — Unicode Encodings

#### (a)

**My answers:**

UTF-8 uses fewer bytes for encoding the training material (English), thanks to it taking single bytes for ASCII characters.

>>> len(list('ABCDEFGabcdefg'.encode('utf-8')))
14
>>> len(list('ABCDEFGabcdefg'.encode('utf-16')))
30
>>> len(list('ABCDEFGabcdefg'.encode('utf-32')))
60

#### (b)

**My answers:**

The function fails when the encoding uses more than one byte for a character. It goes through the encoding byte by byte for the entire string and is not aware of characters encoded by multiple bytes.

>>> decode_utf8_bytes_to_str_wrong("您好".encode("utf-8"))
Traceback (most recent call last):
  File "<python-input-75>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("您好".encode("utf-8"))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python-input-73>", line 2, in decode_utf8_bytes_to_str_wrong
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
                    ~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe6 in position 0: unexpected end of data
>>> "您好".encode("utf-8")
b'\xe6\x82\xa8\xe5\xa5\xbd'
>>> list("您好".encode("utf-8"))
[230, 130, 168, 229, 165, 189]
>>> ("您好".encode("utf-8")).decode("utf-8")
'您好'

#### (c)

**My answers:**

When the first byte starts with 11110 (240-247), utf-8 rules says it needs to look at the following three bytes for encoding. So the byte sequence [240, 1] cannot be decoded. In addition, the following bytes need to start with 10, so [240, 176] fails specifically for the unexpected end of data.

>>> bytes([240,1]).decode('utf-8')
Traceback (most recent call last):
  File "<python-input-103>", line 1, in <module>
    bytes([240,1]).decode('utf-8')
    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf0 in position 0: invalid continuation byte
>>>
>>> bytes([240,176]).decode('utf-8')
Traceback (most recent call last):
  File "<python-input-110>", line 1, in <module>
    bytes([240,176]).decode('utf-8')
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 0-1: unexpected end of data

### train_bpe — BPE Tokenizer Training

**Status:** done

### train_bpe_tinystories — BPE Training on TinyStories

#### (a)

**My answers:**

100s. Longest token is ‘ accomplishment’. Makes sense because it’s a long word. The peak memory usage was ~ 5 Gb.

#### (b)

**My answers:**

The current most time consuming part of tokenizer training is finding the bytepair with maximum count, because it’s performed at each merging iteration and the comparison is performed across all byte pairs. It takes 27.474s for max itself and 25.481s for the lambda function called within max.

### train_bpe_expts_owt — BPE Training on OpenWebText

Had to switch from map to imap_unordered to avoid OOM kill. Also had to switch from list to set to store pretokens of byte pair reverse lookup to make training faster on my laptop. Turns out determining if an element is in a list is O(n), while for a set it’s O(1). This was not evident when training the smaller ministry dataset but for the owt data one byte pair can have many pretokens associated with it so the check for element presence becomes the bottle neck during the merging step.

Finished in 7 hours, 52 minutes, 41 seconds. Your M3 Pro handled the full OpenWebText BPE training with a 32,000-token vocabulary.
- Configuration: 8 worker processes, 500 chunks.
- Maximum resident set size: approximately 5.77 GiB, not the combined simultaneous peak across workers.
- Previous list-based run: still unfinished after 22.5 hours.

#### (a)

**My answers:**

The longest token is ‘’.

#### (b)

**My answers:**

TODO

### tokenizer — Implementing the tokenizer

**Status:** done

### tokenizer_experiments — Experiments with tokenizers

#### (a)

**My answers:**

TODO

#### (b)

**My answers:**

TODO

#### (c)

**My answers:**

TODO

#### (d)

**My answers:**

TODO

### linear — Implementing the linear module

**Status:** done

### embedding — Implement the embedding module

**Status:** done

### rmsnorm — Root Mean Square Layer Normalization

**Status:** done

### positionwise_feedforward — Implement the position-wise feed-forward network

**Status:** done

### rope — Implement RoPE

**Status:** done

### softmax — Implement softmax

**Status:** done

### scaled_dot_product_attention — Implement scaled dot-product attention

**Status:** done

### multihead_self_attention — Implement causal multi-head self-attention

**Status:** done

### transformer_block — Implement the Transformer block

**Status:** done

### transformer_lm — Implementing the Transformer LM

**Status:** done

### transformer_accounting — Transformer LM resource accounting

#### (a)

**My answers:**

- Embedding (vocab_size 50257, d_model 1600)
  - parameters: 50257 x 1600 = 80,411,200
- transformer block x 48
  - layernorm 1
    - parameters: 1600
  - multihead self attention (25 head)
    - q proj
      - parameters: 1600 x 1600 = 2,560,000 
    - k proj
      - parameters: 1600 x 1600 = 2,560,000 
    - v proj
      - parameters: 1600 x 1600 = 2,560,000 
    - output proj
      - parameters: 1600 x 1600 = 2,560,000 
  - layernorm 2
    - parameters: 1600
  - ffn
    - w1
      - parameters: 1600 x 4288 = 6,860,800 
    - w2
      - parameters: 4288 x 1600 = 6,860,800 
    - w3
      - parameters: 1600 x 4288 = 6,860,800 
- final layernorm
  - parameters: 1600
- lm head
  - parameters: 1600 x 50257 = 80,411,200

So the high level breakdown is:

- Embedding: 80,411,200
- Transformer Blocks: (1600 + 2560000 x 4 + 1600 + 6860800 * 3) * 48 = 1,479,628,800 
  - attn vs ffn: 2560000 x 4 = 10,240,000 vs 6,860,800 * 3 = 20,582,400 so roughly 1:2

And total number of parameters is 80,411,200 + 1,479,628,800 + 1600 + 80,411,200 = 1,640,452,800. Each single precision floating point number is 4 bytes, so the memory usage is 1,640,452,800 x 4 = 6,561,811,200 bytes so 6.56 GB.

#### (b)

**My answers:**

For one sample during the forward pass, the FLOPs count:

- Embedding (vocab_size 50257, d_model 1600)
  - this is a lookup, 0 FLOPs
- transformer block x 48
  - layernorm 1
    - 1024 x 1600 x 4 = 6,553,600, approximately. But the question is only asking for matrix multiplications so still 0 here.
  - multihead self attention (25 head):
    - q proj
      - B T d, d d -> B T d so BTd * 2d = 2 x 1600 x 1600 x 1 x 1024 = 5,242,880,000
    - k proj
      - same as above
    - v proj
      - same 
    - output proj
      - same
    - QK_T
      - B h T d_k, B h T d_k -> B h T T so BhTT * 2d_k = 2BdT^2 = 2 x 1 x 1600 x 1024 x 1024 = 3355443200
    - softmax(QK_T) @ V
      - B h T T, B h T d_k -> B h T d_k so BhTd_k * 2T = 2BdT^2 = 2 x 1 x 1600 x 1024 x 1024 = 3355443200
  - layernorm 2
    - see above, count 0 here
  - ffn
    - w1
      - B T d, d d_ff -> B T d_ff so BTd_ff * 2d = 2 x 1600 x 4288 x 1 x 1024 = 1.405×10¹⁰
    - w2
      - B T d_ff, d_ff d -> B T d so BTd * 2d_ff = 1.405×10¹⁰
    - w3
      - the information values, same operations as w1, so also 1.405×10¹⁰
- final layernorm
  - same as before, no matrix multiplication so count 0
- lm head
  - B T d, d vocab_size -> B T vocab_size, so BTvocab_size * 2d = 2 x 1600 * 50257 * 1 * 1024 = 1.647×10¹¹

So total FLOPs is about (5,242,880,000 x 4 + 3355443200 x 2 + 1.405×10¹⁰ x 3) x 48 + 1.647×10¹¹ = 3.517×10¹²

And attn does ~ 66% of the FLOPs of ffn

#### (c)

**My answers:**

ffn

#### (d)

**My answers:**

$$
n_{layers} (8BTd^2 + 4BT^2d + 6BTdd_{ff}) + 2BTdV
$$

- GPT-2 small: d_ff = 2048. 12 * (8 * 1024 * 768^2 + 4 * 1024^2 * 768 + 6 * 1024 * 768 * 2048) + 2 * 1024 * 768 * 50257 = 2.916×10¹¹ 
- GPT-2 medium: d_ff = 2752. 24 * (8 * 1024 * 1024^2 + 4 * 1024^2 * 1024 + 6 * 1024 * 1024 * 2752) + 2 * 1024 * 1024 * 50257 = 8.302×10¹¹ 
- GPT-2 large: d_ff = 3456. 36 * (8 * 1024 * 1280^2 + 4 * 1024^2 * 1280 + 6 * 1024 * 1280 * 3456) + 2 * 1024 * 1280 * 50257 = 1.787×10¹²
- GPT-2 XL: d_ff = 4288. 48 * (8 * 1024 * 1600^2 + 4 * 1024^2 * 1600 + 6 * 1024 * 1600 * 4288) + 2 * 1024 * 1600 * 50257 = 3.517×10¹² 

As the model size increases, attn and ffn takes up more proportions of FLOPs because they scale quadratically with d.

#### (e)

**My answers:**

- GPT-2 XL with long context of 16384: 48 * (8 * 16384 * 1600^2 + 4 * 16384^2 * 1600 + 6 * 16384 * 1600 * 4288) + 2 * 16384 * 1600 * 50257 = 1.336×10¹⁴

The attn FLOPs scales quadratically with T so that components’ proportional FLOPs go up as context length increases. With this context length increase of about 10 fold, FLOPs goes up by almost 2 orders of magnitude. To be specific, it’s the attn token mixing softmax(QK^T/sqrt(d))V that scales quadratically with T so the main contributor.

### cross_entropy — Implement cross-entropy

**Status:** not started

### learning_rate_tuning — Tuning the learning rate

**My answers:**

TODO

### adamw — Implement AdamW

**Status:** not started

### adamw_accounting — Resource accounting for training with AdamW

#### (a)

**My answers:**

TODO

#### (b)

**My answers:**

TODO

#### (c)

**My answers:**

TODO

#### (d)

**My answers:**

TODO

### learning_rate_schedule — Implement cosine learning rate schedule with warmup

**Status:** not started

### gradient_clipping — Implement gradient clipping

**Status:** not started

### data_loading — Implement data loading

**Status:** not started

### checkpointing — Implement model checkpointing

**Status:** not started

### training_together — Put it together

**Status:** not started

### decoding — Decoding

**Status:** not started

### experiment_log — Experiment logging

**My answers:**

TODO

### learning_rate — Tune the learning rate

#### (a)

**My answers:**

TODO

#### (b)

**My answers:**

TODO

### batch_size_experiment — Batch size variations

**My answers:**

TODO

### generate — Generate text

**My answers:**

TODO

### layer_norm_ablation — Remove RMSNorm and train

**My answers:**

TODO

### pre_norm_ablation — Implement post-norm and train

**My answers:**

TODO

### no_pos_emb — Implement NoPE

**My answers:**

TODO

### swiglu_ablation — SwiGLU vs. SiLU

**My answers:**

TODO

### main_experiment — Experiment on OWT

**My answers:**

TODO

### leaderboard — Leaderboard

**My answers:**

TODO
