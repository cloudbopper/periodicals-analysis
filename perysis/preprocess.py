"""
Preprocesses documents for topic modeling

Written using tutorial at https://machinelearningmastery.com/clean-text-machine-learning-python/
"""

import argparse
import glob
import logging
import os
import string

import enchant
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", required=True)
    parser.add_argument("-output_dir", required=True)
    parser.add_argument("-ignore_proper_nouns", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.INFO, filename="%s/preprocess.log" % args.output_dir, format="%(asctime)s: %(message)s")
    logger = logging.getLogger()

    logger.info("Begin preprocessing pipeline")
    pipeline(args, logger)
    logger.info("End preprocessing pipeline")

def pipeline(args, logger):
    """Preprocessing pipeline"""
    dictionary = enchant.Dict("en_US")
    wnl = nltk.WordNetLemmatizer()
    paths = glob.glob("%s/*.txt" % args.input_dir)
    filecount = len(paths)
    for idx, raw_filename in enumerate(paths):
        logger.info("Processing filename %d of %d: %s" % (idx, filecount, raw_filename))
        with open(raw_filename) as raw_file:
            raw_text = raw_file.read()
            # split into tokens
            words = word_tokenize(raw_text)
            # convert to lower case
            words = [word.lower() for word in words]
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            words = [word.translate(table) for word in words]
            # remove remaining words that are not alphabetic
            words = [word for word in words if word.isalpha()]
            # filter out stop words
            stop_words = set(stopwords.words("english"))
            words = [word for word in words if word not in stop_words]
            # lemmatization
            words = [wnl.lemmatize(word) for word in words]
            # spell-check
            if args.ignore_proper_nouns:
                words = [word for word in words if dictionary.check(word)]
            else:
                words = [word for word in words if wn.synsets(word)]
            processed_filename = "%s/%s.txt" % (args.output_dir, os.path.splitext(os.path.basename(raw_filename))[0])
            with open(processed_filename, "w") as processed_file:
                processed_file.write(" ".join(words))


if __name__ == "__main__":
    main()
