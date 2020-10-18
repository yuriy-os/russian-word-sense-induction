import argparse
from typing import Dict, Union

import pandas as pd
from sklearn.metrics import adjusted_rand_score

PandasDataFrame = pd.DataFrame


def gold_predict(df: PandasDataFrame) -> PandasDataFrame:
    """Assigns the gold and predict fields to the data frame. 
    
    Args:
        df: Dataframe with words and their contexts.

    Returns:
        Dataframe with assigned gold and predict fields.

    """

    df = df.copy()

    df["gold"] = df["word"] + "_" + df["gold_sense_id"]
    df["predict"] = df["word"] + "_" + df["predict_sense_id"]

    return df


def ari_per_word_weighted(df: PandasDataFrame) -> Union[float, Dict[str, float]]:
    """Computes the ARI score weighted by the number of sentences per word.
    
    Args:
        df: Dataframe with mutisense words and their contexts.
    
    Returns:
        ARI for every multisense word.

    """

    df = gold_predict(df)

    words = {
        word: (adjusted_rand_score(df_word.gold, df_word.predict), len(df_word))
        for word in df.word.unique()
        for df_word in (df.loc[df["word"] == word],)
    }

    cumsum = sum(ari * count for ari, count in words.values())
    total = sum(count for _, count in words.values())

    assert total == len(df), "please double-check the format of your data"

    return cumsum / total, words


def evaluate(dataset_fpath: str) -> float:
    """Computes the ARI score weighted by the number of sentences per word.
    
    Args:
        dataset_fpath: Path to dataset with predicted senses.
    
    Returns:
        ARI metric for a whole dataset.

    """
    df = pd.read_csv(
        dataset_fpath, sep="\t", dtype={"gold_sense_id": str, "predict_sense_id": str}
    )
    ari, words = ari_per_word_weighted(df)
    print("{}\t{}\t{}".format("word", "ari", "count"))

    for word in sorted(words.keys()):
        print("{}\t{:.6f}\t{:d}".format(word, *words[word]))

    print("\t{:.6f}\t{:d}".format(ari, len(df)))
    return ari


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Evaluate the quality of Word Sense Induction (WSI) "
        "in the shared task http://russe.nlpub.org/2018/wsi/ "
        "based on the Adjusted Rand Index (ARI) metric "
        "between the vectors of the gold standard sense "
        "annotations and the predicted sense annotations. "
        "The participants of the shared "
        'task are supposed to fill the initially void "predict_sense_id" '
        "column of the dataset CSV file. The evaluation script compares "
        "values in this column of the file with the values in the "
        '"gold_sense_id" to compute the ARI metric. The training '
        " datasets in the required format are available in the "
        '"data" directory (train.csv). To run an already pre-filled '
        "baseline system run "
        '"python evaluate.py data/main/wiki-wiki/train.baseline-adagram.csv"',
    )

    parser.add_argument(
        "dataset",
        type=argparse.FileType("r"),
        help="Path to a CSV file with the dataset in the format "
        '"context_id<TAB>word<TAB>gold_sense_id<TAB>predict_sense_id'
        '<TAB>positions<TAB>context". ',
    )
    args = parser.parse_args()
    evaluate(args.dataset)


if __name__ == "__main__":
    main()
