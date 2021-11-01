"""
RAW DATA
"""
import os

os.system('git clone https://github.com/najoungkim/COGS.git')
BASE_DIR = "./COGS/data/"
TRAIN_PATH = str("{}train.tsv".format(BASE_DIR))
TRAIN_100_PATH = str("{}train_100.tsv".format(BASE_DIR))
VALID_PATH = str("{}dev.tsv".format(BASE_DIR))
TEST_PATH = str("{}test.tsv".format(BASE_DIR))
GEN_PATH = str("{}gen.tsv".format(BASE_DIR))


"""
OBTAIN COGS VOCABULARY
"""


def getCOGSParallelData(PATH):
    # read raw file
    with open(PATH) as f:
        data = f.readlines()

    src_vocab, tgt_vocab = set(), set()
    src_lines, tgt_lines, codes = [], [], []

    for line in data:
        source, target, code = line.rstrip("\n").split("\t")
        src_lines.append(source)
        tgt_lines.append(target)
        codes.append(code)
        src_vocab.update(source.split())
        tgt_vocab.update(target.split())
    print("Num sentences in {} = {}".format(PATH, len(src_lines)))
    return src_lines, tgt_lines, list(src_vocab), list(tgt_vocab), codes


# get split-wise data
(
    train_src_lines,
    train_tgt_lines,
    train_src_vocab,
    train_tgt_vocab,
    train_codes,
) = getCOGSParallelData(TRAIN_PATH)
(
    train_100_src_lines,
    train_100_tgt_lines,
    train_100_src_vocab,
    train_100_tgt_vocab,
    train_100_codes,
) = getCOGSParallelData(TRAIN_100_PATH)
(
    valid_src_lines,
    valid_tgt_lines,
    valid_src_vocab,
    valid_tgt_vocab,
    valid_codes,
) = getCOGSParallelData(VALID_PATH)
(
    test_src_lines,
    test_tgt_lines,
    test_src_vocab,
    test_tgt_vocab,
    test_codes,
) = getCOGSParallelData(TEST_PATH)
(
    gen_src_lines,
    gen_tgt_lines,
    gen_src_vocab,
    gen_tgt_vocab,
    gen_codes,
) = getCOGSParallelData(GEN_PATH)

# create a combined vocabulary dictionary
token2id = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2}
index = 3
for vocab in [
    train_src_vocab,
    train_tgt_vocab,
    train_100_src_vocab,
    train_100_tgt_vocab,
    valid_src_vocab,
    valid_tgt_vocab,
    test_src_vocab,
    test_tgt_vocab,
    gen_src_vocab,
    gen_tgt_vocab,
]:
    for i in range(len(vocab)):
        if vocab[i] not in token2id.keys():
            token2id[vocab[i]] = index
            index += 1
print("Total of {} unique tokens in the COGS data".format(index))


# find maximum sequence length in the data
lengths = []
for lines in [test_src_lines, test_tgt_lines]:
    lengths += [len(line.split()) for line in lines]
print("Maximum Sequence Length in the Test data:", max(lengths)+2)
lengths = []
for lines in [gen_src_lines, gen_tgt_lines]:
    lengths += [len(line.split()) for line in lines]
print("Maximum Sequence Length in the Generalization data:", max(lengths)+2)


# Only Take Sentences which are < 200 words and reduce data samples (Due to memory limitations)
tmp_src_lines, tmp_tgt_lines = [], []
for i in range(len(gen_tgt_lines)):
    if len(gen_tgt_lines[i].split()) < 200:
        tmp_src_lines.append(gen_src_lines[i])
        tmp_tgt_lines.append(gen_tgt_lines[i])

gen_tgt_lines = tmp_tgt_lines[::5]
print("Kept only {} lines out of the original {} lines".format(len(tmp_tgt_lines), len(gen_src_lines)))
gen_src_lines = tmp_src_lines[::5]

lengths = []
for lines in [gen_src_lines, gen_tgt_lines]:
    lengths += [len(line.split()) for line in lines]
print("Maximum Sequence Length in the Filtered Generalization data:", max(lengths)+2)