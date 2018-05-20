"""Preprocesses documents for topic modeling"""

import argparse
import glob
import os
import pdb
import pprint
import re

import nltk
from nltk import word_tokenize

def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", required=True)
    parser.add_argument("-output_dir", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    pipeline(args)

def pipeline(args):
    """Preprocessing pipeline"""
    paths = glob.glob("%s/*.txt" % args.input_dir)
    pdb.set_trace()
