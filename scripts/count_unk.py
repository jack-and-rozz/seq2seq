# coding: utf-8
import sys, re, argparse
import collections


#my sources
from utils import common


def main(args):
    UNK_ID = '3'
    m = re.match('(.+)\..ids([0-9]+)(\.?.*)',args.filename)
    seq = common.read_file(args.filename,  do_flatten=True)
    n_vocab = int(m.group(2))
    n_token = len(seq)
    n_unk = seq.count(UNK_ID)

    print(args.filename)
    print(("N_VOCAB: %d" % n_vocab))
    print(("N_TOKEN: %d" % n_token))
    print(("N_UNK: %d" % n_unk))
    print(("Average token incidence: %f" % (n_token / n_vocab)))
    print(("UNK rate: %f %%" % (100.0 * n_unk / n_token)))

if __name__ == "__main__":
    desc = ''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filename', help ='token-ID sequences in "dataset/processed')
    args = parser.parse_args()
    main(args)


