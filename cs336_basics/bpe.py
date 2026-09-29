import multiprocessing
import os
import regex as re
from collections import defaultdict, Counter
from collections.abc import Iterable, Iterator
from functools import partial, reduce
from typing import BinaryIO
from typing import Self
import pickle

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(
    start_end: tuple[int],
    file_path: str,
    special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    start = start_end[0]
    end = start_end[1]
    split_pattern = '|'.join([re.escape(p) for p in special_tokens])
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        if split_pattern != '':
            mini_chunks = re.split(split_pattern, chunk)
        else:
            mini_chunks = [chunk]
    ret = defaultdict(int)
    for mini_chunk in mini_chunks:
        for m in re.finditer(PAT, mini_chunk):
            pretoken = m.group(0).encode("utf-8")
            key = tuple([bytes([t]) for t in pretoken])
            ret[key] += 1
    return ret


def count_bytepair(pretoken: tuple[bytes, ...]) -> dict[tuple[bytes, bytes], int]:
    ret = defaultdict(int)
    if len(pretoken) < 2:
        return dict(ret)
    for i in range(len(pretoken) - 1):
        key = (pretoken[i], pretoken[i+1])
        ret[key] += 1
    return dict(ret)


def update_pretoken(pretoken: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    # updates pretoken merging bytes
    if len(pretoken) < 2:
        return pretoken
    i = 0
    new_list = []
    while i < len(pretoken):
        if i+1 < len(pretoken) and (pretoken[i], pretoken[i+1]) == merge:
            merged = pretoken[i] + pretoken[i+1]
            new_list.append(merged)
            i += 2
        else:
            new_list.append(pretoken[i])
            i += 1
    ret = tuple(new_list)
    return ret


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # pretokenize
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Run pre-tokenization on your chunk and store the counts for each pre-token
        func = partial(pretokenize, file_path=input_path, special_tokens=special_tokens)
        dicts = pool.map(func, zip(boundaries[:-1], boundaries[1:]))
    
    # Maintain 1: pretok_bytes_count_ledger is the most foundemental source of truth, and is updated each iteration
    # initialize pretok_bytes_count_ledger from pretokenize output
    pretok_count = defaultdict(int)
    for d in dicts:
        for k,v in d.items():
            pretok_count[k] += v
    pretok_bytes_count_ledger = dict()
    for k,v in pretok_count.items():
        pretok_bytes_count_ledger[len(pretok_bytes_count_ledger)] = (k, v)
    
    # Maintain 2: bytepair counts
    bp_count = defaultdict(int)

    # Maintain 3: reverse lookup: given byte pair, lookup the pretokein ids it belongs to
    bp_pretok = defaultdict(list)

    # initialize bp_count and bp_pretok
    for pretok_id,v in pretok_bytes_count_ledger.items():
        pretok_bytes, pc = v
        this_bp_count = count_bytepair(pretok_bytes)
        for bp,bc in this_bp_count.items():
            bp_count[bp] += pc * bc
            bp_pretok[bp].append(pretok_id)

    merges = []
    vocab = dict()
    for i in range(256):
        vocab[i] = bytes([i])
    for i in range(len(special_tokens)):
        vocab[256 + i] = special_tokens[i].encode("utf-8")

    # find the bytes to merge, record the merge in merges list, add merged bp to vocab
    # update the three maintained items to reflect the merge
    # keep going until vocab size is reached

    while len(vocab) < vocab_size and len(bp_count) > 0:
        merge_bp = max(bp_count, key=lambda x: (bp_count[x], x[0], x[1]))
        merge_bp_count = bp_count[merge_bp]
        merges.append(merge_bp)
        vocab[len(vocab)] = merge_bp[0] + merge_bp[1]

        # maintain 1 and the subset of ledger that's changed (pre_merge_bytes, post_merge_bytes, pretok_count)
        affect_ledger = dict()
        for pretok_id in bp_pretok[merge_bp]:
            pretok_bytes = pretok_bytes_count_ledger[pretok_id][0]
            pretok_count = pretok_bytes_count_ledger[pretok_id][1]
            updated_pretok_bytes = update_pretoken(pretok_bytes, merge_bp)
            pretok_bytes_count_ledger[pretok_id] = (updated_pretok_bytes, pretok_count)
            affect_ledger[pretok_id] = (pretok_bytes, updated_pretok_bytes, pretok_count)

        # maintain 2 and 3, subtract old affected counts, add new affected counts
        for pretok_id, v in affect_ledger.items():
            old_bp_count = count_bytepair(v[0])
            new_bp_count = count_bytepair(v[1])
            pc = v[2]
            for bp,bc in old_bp_count.items():
                bp_count[bp] -= pc * bc
            for bp,bc in new_bp_count.items():
                bp_count[bp] += pc * bc
                if pretok_id not in bp_pretok[bp]:
                    bp_pretok[bp].append(pretok_id)
        dropped_bps = set([k for k,v in bp_count.items() if v == 0])
        for bp in dropped_bps:
            bp_count.pop(bp, None)
            bp_pretok.pop(bp, None)

        # bp_count = defaultdict(int)
        # bp_pretok = defaultdict(list)
        # for pretok_id,v in pretok_bytes_count_ledger.items():
        #     pretok_bytes, pc = v
        #     this_bp_count = count_bytepair(pretok_bytes)
        #     for bp,bc in this_bp_count.items():
        #         bp_count[bp] += pc * bc
        #         bp_pretok[bp].append(pretok_id)        

    return vocab, merges

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab # id -> bytes, decoding
        reverse_vocab = dict()
        for i, b in vocab.items():
            reverse_vocab[b] = i
        self.reverse_vocab = reverse_vocab # bytes -> id, encoding
        # if special tokens are not in vocab, add them
        if special_tokens != None:
            for st in special_tokens:
                if st.encode("utf-8") not in self.reverse_vocab:
                    self.reverse_vocab[st.encode("utf-8")] = len(vocab)
                    self.vocab[len(vocab)] = st.encode("utf-8")                
        merges_rank = dict()
        for i, pair in enumerate(merges):
            merges_rank[pair] = i
        self.merges_rank = merges_rank
        self.special_tokens = sorted(special_tokens, key=len, reverse=True) if special_tokens != None else []

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) -> Self:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        ret = cls(vocab, merges, special_tokens)
        return ret

    def _encode_pretoken(self, pretoken: tuple[bytes, ...]) -> list[int]:
        candidate_special_token = b"".join(pretoken)
        if candidate_special_token.decode("utf-8") in self.special_tokens:
            return [self.reverse_vocab[candidate_special_token]]
        while len(pretoken) >= 2:
            current_rank = -1
            current_merge = None
            for i in range(len(pretoken) - 1):
                candidate_merge = (pretoken[i], pretoken[i+1])
                if candidate_merge in self.merges_rank:
                    if current_rank < 0 or self.merges_rank[candidate_merge] < current_rank:
                        current_merge = candidate_merge
                        current_rank = self.merges_rank[current_merge]
            if current_rank < 0:
                break
            pretoken = update_pretoken(pretoken, current_merge)
        return [self.reverse_vocab[b] for b in pretoken]

    def _pretokenize(self, text: str) -> list[tuple[bytes, ...]]:
        split_pattern = '|'.join([re.escape(p) for p in self.special_tokens])
        if split_pattern != '':
            mini_chunks = re.split(f"({split_pattern})", text) # keep delimiter
        else:
            mini_chunks = [text]
        ret = list()
        for mini_chunk in mini_chunks:
            if mini_chunk in self.special_tokens:
                ret.append(tuple([bytes([t]) for t in mini_chunk.encode("utf-8")]))
            else:
                for m in re.finditer(PAT, mini_chunk):
                    pretoken = m.group(0).encode("utf-8")
                    ret.append(tuple([bytes([t]) for t in pretoken]))
        return ret
        
    def encode(self, text: str) -> list[int]:
        pretokens = self._pretokenize(text)
        encoded = []
        for pretoken in pretokens:
            encoded.extend(self._encode_pretoken(pretoken))
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            yield from self.encode(s)

    def decode(self, ids: list[int]) -> str:
        ret = b''
        for id in ids:
            ret+= self.vocab[id]
        return ret.decode("utf-8", errors="replace")
