# CS336 Assignment 1 — Answers and experiment notes

Handout: Spring 2026, version 26.0.3. See `cs336_assignment1_basics.pdf` for full prompts and deliverables.

Status options: not started / in progress / ready for review / revised. All answers and results below are intentionally blank. Implementation sections track code and verification; they are not additional written deliverables.

## Working setup

- Hardware: MacBook Pro, M3 Pro, 18 GB unified memory
- Python / PyTorch versions: TODO
- Device used: TODO
- Handout low-resource adaptations used: TODO
- Experiment artifacts directory: TODO

## Progress

- [ ] `unicode1`
- [ ] `unicode2`
- [ ] `train_bpe`
- [ ] `train_bpe_tinystories`
- [ ] `train_bpe_expts_owt`
- [ ] `tokenizer`
- [ ] `tokenizer_experiments`
- [ ] `linear`
- [ ] `embedding`
- [ ] `rmsnorm`
- [ ] `positionwise_feedforward`
- [ ] `rope`
- [ ] `softmax`
- [ ] `scaled_dot_product_attention`
- [ ] `multihead_self_attention`
- [ ] `transformer_block`
- [ ] `transformer_lm`
- [ ] `transformer_accounting`
- [ ] `cross_entropy`
- [ ] `learning_rate_tuning`
- [ ] `adamw`
- [ ] `adamw_accounting`
- [ ] `learning_rate_schedule`
- [ ] `gradient_clipping`
- [ ] `data_loading`
- [ ] `checkpointing`
- [ ] `training_together`
- [ ] `decoding`
- [ ] `experiment_log`
- [ ] `learning_rate`
- [ ] `batch_size_experiment`
- [ ] `generate`
- [ ] `layer_norm_ablation`
- [ ] `pre_norm_ablation`
- [ ] `no_pos_emb`
- [ ] `swiglu_ablation`
- [ ] `main_experiment`
- [ ] `leaderboard`

## Responses

### unicode1 — Understanding Unicode

**Status:** revised

#### (a)

chr(0) returns the null character.

#### (b)

chr(0).__repr__() returns `'\x00'`, and print(chr(0)) prints the invisible null character followed by a new line.

#### (c)

The representation of chr(0) is the string of its hexadecimal representation `\x00`. When chr(0) the null character is embedded in a string and printed out, the printed outcome shows nothing visible at the place where chr(0) the null character appears. However, it does take up one character in length. Also, ord(chr(0)) is the unicode code point of chr(0) the null character, and is the integer 0.

### unicode2 — Unicode Encodings

**Status:** not started

#### (a)

**My answer:**

UTF-8 uses fewer bytes for encoding the training material (English), thanks to it taking single bytes for ASCII characters.

**Evidence / calculations / observations:**

>>> len(list('ABCDEFGabcdefg'.encode('utf-8')))
14
>>> len(list('ABCDEFGabcdefg'.encode('utf-16')))
30
>>> len(list('ABCDEFGabcdefg'.encode('utf-32')))
60

#### (b)

**My answer:**

The function fails when the encoding uses more than one byte for a character. It goes through the encoding byte by byte for the entire string and is not aware of characters encoded by multiple bytes.

**Evidence / calculations / observations:**

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

**My answer:**

When the first byte starts with 11110 (240-247), utf-8 rules says it needs to look at the following three bytes for encoding. So the byte sequence [240, 1] cannot be decoded. In addition, the following bytes need to start with 10, so [240, 176] fails specifically for the unexpected end of data.

**Evidence / calculations / observations:**

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

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### train_bpe — BPE Tokenizer Training

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### train_bpe_tinystories — BPE Training on TinyStories

**Status:** not started

#### (a)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (b)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### train_bpe_expts_owt — BPE Training on OpenWebText

**Status:** not started

#### (a)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (b)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### tokenizer — Implementing the tokenizer

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### tokenizer_experiments — Experiments with tokenizers

**Status:** not started

#### (a)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (b)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (c)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (d)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### linear — Implementing the linear module

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### embedding — Implement the embedding module

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### rmsnorm — Root Mean Square Layer Normalization

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### positionwise_feedforward — Implement the position-wise feed-forward network

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### rope — Implement RoPE

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### softmax — Implement softmax

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### scaled_dot_product_attention — Implement scaled dot-product attention

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### multihead_self_attention — Implement causal multi-head self-attention

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### transformer_block — Implement the Transformer block

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### transformer_lm — Implementing the Transformer LM

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### transformer_accounting — Transformer LM resource accounting

**Status:** not started

#### (a)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (b)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (c)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (d)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (e)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### cross_entropy — Implement cross-entropy

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### learning_rate_tuning — Tuning the learning rate

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### adamw — Implement AdamW

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### adamw_accounting — Resource accounting for training with AdamW

**Status:** not started

#### (a)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (b)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (c)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (d)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### learning_rate_schedule — Implement cosine learning rate schedule with warmup

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### gradient_clipping — Implement gradient clipping

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### data_loading — Implement data loading

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### checkpointing — Implement model checkpointing

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### training_together — Put it together

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### decoding — Decoding

**Status:** not started

- Implementation file / function: TODO
- Tests run and results: TODO
- Remaining questions: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### experiment_log — Experiment logging

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### learning_rate — Tune the learning rate

**Status:** not started

#### (a)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

#### (b)

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### batch_size_experiment — Batch size variations

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### generate — Generate text

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### layer_norm_ablation — Remove RMSNorm and train

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### pre_norm_ablation — Implement post-norm and train

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### no_pos_emb — Implement NoPE

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### swiglu_ablation — SwiGLU vs. SiLU

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### main_experiment — Experiment on OWT

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO

### leaderboard — Leaderboard

**Status:** not started

**My answer:**

TODO

**Evidence / calculations / observations:**

TODO

**Experiment record:**

- Configuration / seed / dataset: TODO
- Device / elapsed time / training budget: TODO
- Metrics / curves / generated text: TODO
- Artifact paths: TODO
- Deviations from the handout, if any: TODO

**Questions for review:**

TODO

**Review notes / revision:**

TODO
