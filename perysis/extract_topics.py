"""
Extracts topic models from historical periodicals
"""

from __future__ import print_function
import argparse
import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from .load_documents import load_documents, DATA, YEAR

# pylint: disable = invalid-name

def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_dir", help="input directory containing documents, meta-data", required=True)
    parser.add_argument("-output_dir", help="output directory for extracted models", required=True)
    parser.add_argument("-num_features", help="maximum number of features (bounded by vocabulary size)",
                        type=int, default=2000)
    parser.add_argument("-num_topics", help="number of topics", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(level=logging.INFO, filename="%s/extract_topics.log" % args.output_dir, format="%(asctime)s: %(message)s")
    logger = logging.getLogger()

    pipeline(args, logger)


def pipeline(args, logger):
    """extract_topics pipeline"""
    logger.info("Begin extract_topics pipeline")
    documents = load_documents(args.input_dir)
    tf_vectorizer, dtm_tf, dtm_tfidf = gen_document_term_matrices(args, documents[DATA])
    gen_models(args, logger, documents, tf_vectorizer, dtm_tf, dtm_tfidf)
    logger.info("End extract_topics pipeline")


def gen_models(args, logger, documents, tf_vectorizer, dtm_tf, dtm_tfidf):
    """Generates and outputs topic models"""
    # pylint: disable = too-many-arguments
    models = []

    # Non-negative Matrix Factorization (NMF) model with Frobenius norm
    logger.info("Generating NMF model with Frobenius norm")
    nmf_frobenius = NMF(n_components=args.num_topics, random_state=1, alpha=.1, l1_ratio=.3)
    W = nmf_frobenius.fit_transform(dtm_tfidf)
    T = gen_topic_temporal_dists(documents, nmf_frobenius, tf_vectorizer, dtm_tf)
    model = (nmf_frobenius, W, T)
    with open("%s/nmf_frobenius.pkl" % args.output_dir, "wb") as model_file:
        pickle.dump(model, model_file)
    models.append(model)

    # NMF model with Kullback-Leibler (KL) divergence
    logger.info("Generating NMF model with KL divergence")
    nmf_kl = NMF(n_components=args.num_topics, random_state=1, beta_loss="kullback-leibler", solver="mu", alpha=.1, l1_ratio=.3)
    W = nmf_kl.fit_transform(dtm_tfidf)
    T = gen_topic_temporal_dists(documents, nmf_kl, tf_vectorizer, dtm_tf)
    model = (nmf_kl, W, T)
    with open("%s/nmf_kl.pkl" % args.output_dir, "wb") as model_file:
        pickle.dump(model, model_file)
    models.append(model)

    # Latent Dirichlet Allocation (LDA) model with TF
    # http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    logger.info("Generating LDA model with TF")
    lda_tf = LatentDirichletAllocation(n_components=args.num_topics, max_iter=100, random_state=1, learning_method="batch")
    W = lda_tf.fit_transform(dtm_tf)
    T = gen_topic_temporal_dists(documents, lda_tf, tf_vectorizer, dtm_tf)
    model = (lda_tf, W, T)
    with open("%s/lda_tf.pkl" % args.output_dir, "wb") as model_file:
        pickle.dump(model, model_file)
    models.append(model)

    return models


def gen_document_term_matrices(args, data):
    """Generates document-term matrices"""

    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    tf_vectorizer = CountVectorizer(stop_words="english",
                                    max_features=args.num_features,
                                    max_df=0.95,
                                    min_df=2)
    dtm_tf = tf_vectorizer.fit_transform(data)
    with open("%s/dtm_tf.pkl" % args.output_dir, "wb") as dtm_file:
        pickle.dump(tf_vectorizer, dtm_file)
        pickle.dump(dtm_tf, dtm_file)

    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    tfidf_vectorizer = TfidfVectorizer(**tf_vectorizer.get_params())
    dtm_tfidf = tfidf_vectorizer.fit_transform(data)
    with open("%s/dtm_tfidf.pkl" % args.output_dir, "wb") as dtm_file:
        pickle.dump(tfidf_vectorizer, dtm_file)
        pickle.dump(dtm_tfidf, dtm_file)
    return tf_vectorizer, dtm_tf, dtm_tfidf


def gen_topic_temporal_dists(documents, model, tf_vectorizer, dtm_tf):
    """Generates temporal distributions for topics"""
    years = documents[YEAR]
    yrange = range(years.min(), years.max() + 1, 1)
    df = pd.DataFrame(np.zeros((len(yrange), dtm_tf.shape[1]), dtype=np.int64), columns=tf_vectorizer.get_feature_names(), index=yrange)
    for idx in documents.index:
        year = years[idx]
        df.loc[year] += dtm_tf[idx].toarray().flatten()

    topics_by_year = np.dot(model.components_, df.T)
    return topics_by_year


if __name__ == "__main__":
    main()
