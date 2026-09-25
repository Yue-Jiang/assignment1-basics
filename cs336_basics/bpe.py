import argparse
import multiprocessing
import os
import regex as re
from collections import defaultdict, Counter
from functools import partial, reduce
from typing import BinaryIO

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Byte-pair encoding exercise"
    )
    parser.add_argument(
        "-i", "--input-path",
        type=str,
        help="Path to a text file with BPE tokenizer training data"
    )
    parser.add_argument(
        "-v", "--vocab-size",
        type=int,
        help="A positive integer that defines the maximum final vocabulary size"
    )
    parser.add_argument(
        "-s", "--special-tokens",
        type=list[str],
        help="A list of strings to add to the vocabulary"
    )
    return parser.parse_args()

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

        # maintain 1
        for pretok_id,v in pretok_bytes_count_ledger.items():
            if pretok_id in bp_pretok[merge_bp]:
                pretok_bytes = pretok_bytes_count_ledger[pretok_id][0]
                pretok_count = pretok_bytes_count_ledger[pretok_id][1]
                pretok_bytes_count_ledger[pretok_id] = (update_pretoken(pretok_bytes, merge_bp), pretok_count)

        # maintain 2 and 3
        # TODO: only update based on pretokens where merge happened
        bp_count = defaultdict(int)
        bp_pretok = defaultdict(list)
        for pretok_id,v in pretok_bytes_count_ledger.items():
            pretok_bytes, pc = v
            this_bp_count = count_bytepair(pretok_bytes)
            for bp,bc in this_bp_count.items():
                bp_count[bp] += pc * bc
                bp_pretok[bp].append(pretok_id)        

    return vocab, merges
    

if __name__ == "__main__":
    args = parse_arguments()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens, 4)