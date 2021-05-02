""" Evaluate the baselines on ROUGE"""
import json
import os
from os.path import join, exists
import argparse
from evaluate import eval_rouge

dec_dir = args.decode_dir
ref_dir = args.ref_dir

def main(dec_dir, ref_dir):
    dec_pattern = r'test-(\d+).txt'
    ref_pattern = 'test-#ID#.txt'
    output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
    metric = 'rouge'
    print(output)
    with open(join(dec_dir, '{}.txt'.format(metric)), 'w') as f:
        f.write(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate summarization models')
    parser.add_argument('--decode_dir', action='store', required=True,
                        help='directory of decoded summaries')
    parser.add_argument('--ref_dir', action='store', required=True,
                        help='directory of reference summaries')
    args = parser.parse_args()
    main(dec_dir, ref_dir)
