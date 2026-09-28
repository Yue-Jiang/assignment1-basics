import argparse
import pickle
from bpe import train_bpe

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
        type=str,
        nargs="+",
        help="A list of strings to add to the vocabulary"
    )
    parser.add_argument(
        "-n", "--num-threads",
        type=int,
        help="Number of threads to use",
        default=1
    )
    parser.add_argument(
        "-o", "--output-prefix",
        help="Output prefix for pickle of vocab, merges. The saved files will be {output-prefix}-vocab.pkl and {output-prefix}-merges.pkl"
    )
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_arguments()
    vocab, merges = train_bpe(args.input_path, args.vocab_size, args.special_tokens, args.num_threads)
    with open(f"{args.output_prefix}-vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{args.output_prefix}-merges.pkl", "wb") as f:
        pickle.dump(merges, f)

# /usr/bin/time -l uv run python cs336_basics/train_bpe.py -i data/TinyStoriesV2-GPT4-train.txt -v 10000 -s "<|endoftext|>" -n 8 -o output/TinyStoriesV2-GPT4-trained-bpe
# uv run python -m cProfile -o bpe.prof cs336_basics/train_bpe.py -i data/TinyStoriesV2-GPT4-train.txt -v 10000 -s "<|endoftext|>" -n 8 -o output/TinyStoriesV2-GPT4-trained-bpe
# uv run --with memory-profiler mprof run --include-children --interval 0.1 --output bpe-memory.dat python cs336_basics/train_bpe.py -i data/TinyStoriesV2-GPT4-train.txt -v 10000 -s "<|endoftext|>" -n 8 -o output/TinyStoriesV2-GPT4-trained-bpe
# uv run --with memory-profiler --with matplotlib mprof plot bpe-memory.dat
