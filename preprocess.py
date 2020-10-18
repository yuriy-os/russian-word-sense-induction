import re
from typing import Optional, List

import nltk
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from ufal.udpipe import Model, Pipeline
from nltk.corpus import stopwords

nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")

UDPIPE_PATH = "./udpipe/udpipe_syntagrus.model"

NumpyArray = np.array
UDPipePipeline = Pipeline


def clean_token(token: str, misc: str) -> Optional[str]:
    """Cleans whitespace, filters technical information

    Args:
        token: Token.
        misc: Contents of the "MISC" field in CONLLU format
    
    Returns:
        Clean token.

    """
    out_token = token.strip().replace(" ", "")
    if token == "Файл" and "SpaceAfter=No" in misc:
        return None
    return out_token


def clean_lemma(lemma: str, pos: str) -> str:
    """Cleans whitespace and special symbols
    
    Args:
        lemma: Raw token lemma.
        pos: Lemma POS.
    
    Returns:
        Clean lemma.

    """
    out_lemma = lemma.strip().replace(" ", "").replace("_", "").lower()
    if pos != "PUNCT":
        if out_lemma.startswith("«") or out_lemma.startswith("»"):
            out_lemma = "".join(out_lemma[1:])
        if out_lemma.endswith("«") or out_lemma.endswith("»"):
            out_lemma = "".join(out_lemma[:-1])
        if (
            out_lemma.endswith("!")
            or out_lemma.endswith("?")
            or out_lemma.endswith(",")
            or out_lemma.endswith(".")
        ):
            out_lemma = "".join(out_lemma[:-1])
    return out_lemma


def get_udpipe_tags(udpipe_result: List[str]) -> List[List[str]]:
    """Converts raw string from UDPipe to convenient list of UDPipe tags.

    Args:
        udpipe_result: Raw string from UDPipe.
    
    Returns:
        List of UDPipe tags

    """
    udpipe_result = [l for l in udpipe_result.split("\n") if not l.startswith("#")]
    return [w.split("\t") for w in udpipe_result if w]


def preprocess_dataset(
    data: NumpyArray,
    udpipe_pipeline: UDPipePipeline,
    use_lemma: bool = True,
    use_pos: bool = True,
    remove_punct: bool = True,
    remove_stopwords: bool = True,
) -> List[str]:
    """Preprocesses dataset.

    Dataset preprocessing includes:
    - removing numbers and punctuation;
    - lemmatization;
    - removing stopwords;
    - POS-tagging.
    
    Args:
        data: raw strings.
        udpipe_pipeline: UDPipe pipeline.
        use_lemma: whether generate lemmas or not.
        use_pos: whether generate POS tags or not.
        remove_punct: whether remove punctuation or not.
        remove_stopwords: whether remove stopwords or not.
    
    Returns:
        Preprocessed dataset.

    """
    processed_data = []
    for text in tqdm(data):
        if remove_punct:
            text = text.replace("-", " ")
            text = re.sub(r"[^\w\s]", "", text)
        text_info = get_udpipe_tags(udpipe_pipeline.process(text))
        processed_text = []
        for info in text_info:
            (word_id, token, lemma, pos, xpos, feats, head, deprel, deps, misc) = info
            lemma = clean_lemma(lemma, pos)
            token = clean_token(token, misc).lower()
            if remove_stopwords and lemma in russian_stopwords:
                continue
            text_representation = []
            if use_lemma:
                text_representation.append(lemma)
            else:
                text_representation.append(token)
            if use_pos:
                text_representation.append(pos)
            processed_text.append("_".join(text_representation))
        processed_data.append(" ".join(processed_text))
    return processed_data


if __name__ == "__main__":
    print("Loading UDPipe model")
    udpipe_model = Model.load(UDPIPE_PATH)
    udpipe_pipeline = Pipeline(
        udpipe_model, "tokenize", Pipeline.DEFAULT, Pipeline.DEFAULT, "conllu"
    )
    print("UDPipe model loaded")
    print("Loading train dataset")
    dataset = pd.read_csv("./data/main/active-dict/train.csv", sep="\t")
    print("Preprocessing train dataset")
    dataset["word"] = preprocess_dataset(
        dataset["word"].values, udpipe_pipeline, use_pos=False, remove_stopwords=False
    )
    dataset["context"] = preprocess_dataset(
        dataset["context"].values,
        udpipe_pipeline,
        use_pos=False,
        remove_stopwords=False,
    )
    print("Saving preprocessed dataset")
    dataset.to_csv("./data/main/active-dict/preprocessed_train_copy.csv", sep="\t")
