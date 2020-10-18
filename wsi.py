import json
import warnings
from argparse import ArgumentParser
from os import path

import gensim
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.cluster import AgglomerativeClustering, AffinityPropagation
from sklearn.decomposition import PCA

from utils import embed, save, get_chi2_scores, get_tf_idf_coeffs
from evaluate import evaluate

warnings.filterwarnings('ignore')


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", help="Path to config file", required=True)
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    modelfile = config["embeddings"]["path"]
    print(f"Loading embeddings from {modelfile} ...")
    if modelfile.endswith(".bin.gz") or modelfile.endswith(".bin"):
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=True
        )
    elif modelfile.endswith(".vec.gz"):
        embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
            modelfile, binary=False
        )
    else:
        embedding_model = gensim.models.Word2Vec.load(modelfile)
    embedding_model.init_sims(replace=True)
    print(f"Done")

    dataset_path = config["dataset"]["path"]
    word_column = config["dataset"]["query_word_column"]
    context_column = config["dataset"]["context_column"]
    gold_sense_colunmn = config["dataset"]["gold_sense_column"]
    print(f"Loading dataset from {dataset_path} ...")
    dataset = pd.read_csv(dataset_path, sep="\t", encoding="utf-8")
    print("Done")

    predicted = []
    for query in tqdm(dataset[word_column].unique()):
        print(f"Word: {query}")
        subset = dataset[dataset[word_column] == query]
        contexts = subset[context_column].values
        gold_senses = subset[gold_sense_colunmn].values

        # Initializing matrix for vector representations of context
        matrix = np.empty((subset.shape[0], embedding_model.vector_size))
        row_idx = 0

        chi2_weights = None
        tf_idf_weights = None
        freq_weights = None

        if config["weights"]["use_chi2"]:
            # Compute chi2 scores and use them as weights for word vectors
            chi2_scores, count_vect_vocab = get_chi2_scores(contexts, gold_senses)
        if config["weights"]["use_tf_idf"]:
            # Compute TF-IDF coeffs and use them as weights for word vectors
            tf_idf_coeffs, tf_idf_vocab = get_tf_idf_coeffs(contexts)

        for context_idx, line in enumerate(subset.iterrows()):
            query_context = line[1][context_column]
            bow = [word for word in query_context.split() if word != query]
            # Pair each word from context with corresponding weight
            if config["weights"]["use_chi2"]:
                chi2_weights = {
                    token: chi2_scores[count_vect_vocab[token]] for token in bow
                }
            if config["weights"]["use_tf_idf"]:
                tf_idf_weights = {
                    token: tf_idf_coeffs[context_idx][0, tf_idf_vocab[token]]
                    for token in bow
                }
            if config["weights"]["use_freq"]:
                freq_weights = {
                    token: get_freq_weights(token, embedding_model) for token in bow
                }

            # Create vectors for each context sentence
            embedding = embed(
                bow,
                embedding_model,
                chi2_weights=chi2_weights,
                tf_idf_weights=tf_idf_weights,
                freq_weights=freq_weights,
                chi2_weights_pow=config["weights"]["chi2"]["power"],
                tf_idf_weights_pow=config["weights"]["tf_idf"]["power"],
                freq_weights_pow=config["weights"]["freq"]["power"],
            )
            matrix[row_idx, :] = embedding
            row_idx += 1

        # Reduce dimensionality of context vectors
        pca_decomposed_embeddings = PCA(
            n_components=config["dim_reduction"]["pca"]["n_components"]
        ).fit_transform(matrix)

        clusterable_embedding = UMAP(
            n_neighbors=config["dim_reduction"]["umap"]["n_neighbors"],
            min_dist=config["dim_reduction"]["umap"]["min_dist"],
            n_components=config["dim_reduction"]["umap"]["n_components"],
            random_state=config["dim_reduction"]["umap"]["random_state"],
        ).fit_transform(pca_decomposed_embeddings)

        # Use already defined number of clusters ...
        if config["clustering"]["clusters_predefined"]:
            n_clusters = config["clustering"]["n_clusters_predefined"]
        else:
            # ... or try to infer it using another clustering algorithm
            clustering = AffinityPropagation(
                damping=config["clustering"]["aff_prop"]["damping"], preference=None,
            ).fit(clusterable_embedding)
            n_clusters = len(clustering.cluster_centers_indices_)
            if n_clusters < 1:
                print("Fallback to 1 cluster!")
                n_clusters = 1
            elif n_clusters == len(contexts):
                print("Fallback to 4 clusters!")
                n_clusters = 4

        if config["clustering"]["use_hdbscan"]:
            clustering = HDBSCAN(
                min_samples=config["clustering"]["hdbscan"]["min_samples"],
                min_cluster_size=config["clustering"]["hdbscan"]["min_cluster_size"],
            ).fit_predict(clusterable_embedding)
        else:
            # Use either manualy defined or inferred using another clsustering algorithm
            # number of clusters to finally cluster our contexts
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=config["clustering"]["agg_clust"]["linkage"],
            ).fit(clusterable_embedding)

        cur_predicted = (
            clustering.tolist()
            if config["clustering"]["use_hdbscan"]
            else clustering.labels_.tolist()
        )
        predicted += cur_predicted
        gold = subset.gold_sense_id
        print("Gold clusters:", len(set(gold)))
        print("Predicted clusters:", len(set(cur_predicted)))
    dataset.predict_sense_id = predicted
    fname = path.splitext(path.basename(dataset_path))[0]
    evaluate(save(dataset, fname))


if __name__ == "__main__":
    main()
