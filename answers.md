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

Had to switch from map to imap_unordered to avoid OOM kill.

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

**Status:** not started

### embedding — Implement the embedding module

**Status:** not started

### rmsnorm — Root Mean Square Layer Normalization

**Status:** not started

### positionwise_feedforward — Implement the position-wise feed-forward network

**Status:** not started

### rope — Implement RoPE

**Status:** not started

### softmax — Implement softmax

**Status:** not started

### scaled_dot_product_attention — Implement scaled dot-product attention

**Status:** not started

### multihead_self_attention — Implement causal multi-head self-attention

**Status:** not started

### transformer_block — Implement the Transformer block

**Status:** not started

### transformer_lm — Implementing the Transformer LM

**Status:** not started

### transformer_accounting — Transformer LM resource accounting

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

#### (e)

**My answers:**

TODO

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
