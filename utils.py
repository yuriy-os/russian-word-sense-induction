from typing import Optional, List, Dict, Union

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


GensimKeyedVectors = KeyedVectors
NumpyArray = np.array


def embed(
    text: List[str],
    model: GensimKeyedVectors,
    chi2_weights: Optional[Dict[str, float]] = None,
    tf_idf_weights: Optional[Dict[str, float]] = None,
    freq_weights: Optional[Dict[str, float]] = None,
    chi2_weights_pow: float = 1,
    tf_idf_weights_pow: float = 1,
    freq_weights_pow: float = 1,
) -> NumpyArray:
    """Embeds tokens into a single vector by averaging all token vectors.
    
    Args:
        text: Tokenized sentence.
        model: Embedding model in Gensim format.
        chi2_weights: Chi2 weights for each word in context.
        tf_idf_weights: TF-IDF weights for each word in context.
        freq_weights: Frequency weights for each word in context.
        chi2_weights_pow: Chi2 weights power.
        tf_idf_weights_pow: TF-IDF weights power.
        freq_weights_pow: Frequency weights power.
    
    Returns:
        Averaged (and weighted) vector representing sentence.
    
    """
    chi2_weight = 1
    tf_idf_weight = 1
    freq_weight = 1

    words = [w for w in text if w in model.vocab]
    lexicon = list(set(words))
    lw = len(lexicon)
    if lw == 0:
        print(f"Empty lexicon in {text}")
        return np.zeros(model.vector_size)
    vectors = np.zeros((lw, model.vector_size))
    for i, word in enumerate(lexicon):
        if chi2_weights:
            chi2_weight = np.power(chi2_weights[word], chi2_weights_pow)
        if tf_idf_weights:
            tf_idf_weight = np.power(tf_idf_weights[word], tf_idf_weights_pow)
        if freq_weights:
            freq_weight = np.power(freq_weights[word], freq_weights_pow)
        vectors[i, :] = (
            model[word] * chi2_weight * tf_idf_weight * freq_weight
        )  # Adding word and its vector to matrix
    context_embedding = np.sum(
        vectors, axis=0
    )  # Computing sum of all vectors in the document
    context_embedding = np.divide(context_embedding, lw)  # Computing average vector
    return context_embedding


def get_freq_weights(
    word: str,
    model: GensimKeyedVectors,
    a: float = np.float_power(10, -3),
    wcount: int = 250000000,
) -> float:
    """Weight word based on its frequency

    Some Gensim models are able to retain information about frequency 
    distribution of particular words. 
    
    We are using this information in order to weight words based on their frequency.

    Args:
        word: Raw word.
        model: Embedding model in Gensim format.
        a: Smoothing coefficient.
        wcount: The number of words in the corpus on which the model was trained (by default - the number of words in the RNC).
        
    Returns:
        Word weight (rare words get bigger weights).

    """
    if word in model:
        prob = model.vocab[word].count / wcount
        weight = a / (a + prob)
        return weight
    return 1


def get_chi2_scores(
    contexts: NumpyArray, gold_senses: NumpyArray
) -> Union[NumpyArray, Dict[str, int]]:
    """Calculates chi2 scores for each word in context.

    Args:
        contexts: List of context sentences.
        gold_senese: List of gold(train) senses.

    Returns:
        Chi2 scores for each word in context and vectorizer 
        vocabulary with word indexes.

    """
    count_vectorizer = CountVectorizer(
        lowercase=False, tokenizer=lambda text: text.split()
    )
    count_vect = count_vectorizer.fit_transform(contexts)
    return chi2(count_vect, gold_senses)[0], count_vectorizer.vocabulary_


def get_tf_idf_coeffs(contexts: NumpyArray) -> Union[NumpyArray, Dict[str, int]]:
    """Calculates TF-IDF coefficients for each word in context.

    Args:
        contexts: List of context sentences.

    Returns:
        TF-IDF coefficients for each word in context and vectorizer 
        vocabulary with word indexes.

    """
    tfidf = TfidfVectorizer(lowercase=False, tokenizer=lambda text: text.split())
    return tfidf.fit_transform(contexts).todense(), tfidf.vocabulary_


def save(df, corpus):
    """Saves dataset with predicted senses to CSV file.
    
    Args:
        df: Dataframe with mutisense words and their contexts.
        corpus: Name of the original file.
    Returns:
        Path to saved CSV file with predicted senses.

    """
    output_fpath = corpus + "_predictions.csv"
    df.to_csv(output_fpath, sep="\t", encoding="utf-8", index=False)
    print("Generated dataset: {}".format(output_fpath))
    return output_fpath
