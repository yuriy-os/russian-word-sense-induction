# russian-word-sense-induction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PqSbjrHX8yCs4KUY2igpQCJGeKcvYkae?usp=sharing)

## Task
*From the competition's [website](https://nlpub.github.io/russe-wsi-kit/):*

You are given a word, e.g. `"замок"` and a bunch of text fragments (aka "contexts") where this word occurs, e.g. `"замок владимира мономаха в любече"` and `"передвижению засова ключом в замке"`. You need to cluster these contexts in the (unknown in advance) number of clusters which correspond to various senses of the word. In this example, you want to have two groups with the contexts of the `"lock"` and the `"castle"` senses of the word `"замок"`.

## Data
For this task I only used `active-dict` dataset.

The dataset looked like this (e.g. for Russian word `двигатель (engine)`:

| context_id | word      | gold_sense_id | predict_sense_id | positions | context                                                      |
| ---------- | --------- | ------------- | ---------------- | --------- | ------------------------------------------------------------ |
| 37         | двигатель | 1             |                  | 0-10      | Двигатель взревел, и машина резко рванулась с места          |
| 38         | двигатель | 1             |                  | 10-20     | Если один двигатель откажет, остальные дотянут самолет до места посадки |
| 43         | двигатель | 2             |                  | 7-17      | Труд – двигатель культуры                                    |
| 46         | двигатель | 2             |                  | 57-67     | Он откроет тебе широкий кредит, а кредит, братец ты мой, двигатель торговли и коммерции |

Each training data entry contains a target word (the `word` column) and a context that represents the word (the `context` column). The `gold_sense_id` contains the correct sense identifier.

## My approach
I took a fairly simple [approach](https://arxiv.org/ftp/arxiv/papers/1803/1803.05795.pdf):

1. Embed every lemmatized word in `context` using pre-trained embeddings.
2. Get context vector by averaging all word vectors from which this context consists of.
3. Cluster context vector representations into different groups representing the "sense" of ambiguous `word`.

## Preprocessing
The data was preprocessed as following:

* removal of punctuation marks, numbers from `context`;
* lemmatization of each token in `context`;
* removal of stop words from `context`;
* POS tagging of each lemma in `context`.

## Embeddings
I used [RusVectores](https://rusvectores.org/ru/models/) for this task.

For the `active-dict` dataset the embedding model trained on the RNC - **ruscorpora_upos_cbow_300_20_2019** showed the best result.

For the `active-rutenten` dataset (which mainly consisted of texts obtained by web scraping), a model trained on the Areneum corpus **araneum_upos_skipgram_300_2_2018** was used. This embeddings were trained using texts obtained from sites in the .ru and .рф domains.

Also I tried to use sentence-level embeddings like [USE](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) and [RusVectores ELMo](https://rusvectores.org/ru/models/), but the clustering results were either comparable to W2V or worse.

### Vector weights
The vector of each word in `context` was weighted by varying weights (**chi2 coefficients, tf_idf, weights based on word frequencies**). This approach [showed](http://www.dialog-21.ru/media/4538/arefyevn_ermolaevp_panchenkoa.pdf) promising results.

## Clustering

## Results

| Dataset/ARI  | AdaGram  | Trivial (random-1-3)  | My approach  |
|---|---|---|---|
| active-dict **train**  |0.159930|0.014134|**0.240510**|
| active-dict **test**  |0.161189|-0.006112|**0.237777**|
| active-rutenten **train** |0.195162|0.000677|**0.216266** (3 clusters)|
