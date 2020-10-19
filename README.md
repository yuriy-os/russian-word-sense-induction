# russian-word-sense-induction

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PqSbjrHX8yCs4KUY2igpQCJGeKcvYkae?usp=sharing)

## Task
*From the competition's [website](https://nlpub.github.io/russe-wsi-kit/):*

> You are given a word, e.g. `"замок"` and a bunch of text fragments (aka "contexts") where this word occurs, e.g. `"замок владимира мономаха в любече"` and `"передвижению засова ключом в замке"`. You need to cluster these contexts in the (unknown in advance) number of clusters that correspond to various senses of the word. In this example, you want to have two groups with the contexts of the `"lock"` and the `"castle"` senses of the word `"замок"`.

## Data
For this task, I only used the `active-dict` dataset.

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
The data were preprocessed as follows:

* removal of punctuation marks, numbers from `context`;
* lemmatization of each token in `context`;
* removal of stop words from `context`;
* POS tagging of each lemma in `context`.

Very often the context consisted of several sentences.

Some of them have been cut off during compiling of the original dataset, so we were left with this, e.g.:

> японского – то милое и приятное. В нашем случае – яркое, пестрое и разнообразное. В **альбом** вошло 13 песен и два видеотрекы как бонус – на песни «Радио твое» и «Не напишу»

Based on the hypothesis that the meaning of a word is determined by its environment, I decided to remove sentences from the context that do not initially contain the word for which we are trying to determine its meaning.

After this kind of preprocessing, the above sentence will look like this:

> *<s>японского – то милое и приятное. В нашем случае – яркое, пестрое и разнообразное.</s> В **альбом** вошло 13 песен и два видеотрекы как бонус – на песни «Радио твое» и «Не напишу»*

## Embeddings
I used [RusVectores](https://rusvectores.org/ru/models/) for this task.

For the `active-dict` dataset the embedding model trained on the RNC - **ruscorpora_upos_cbow_300_20_2019** showed the best result.

For the `active-rutenten` dataset (which mainly consisted of texts obtained by web scraping), a model trained on the Areneum corpus **araneum_upos_skipgram_300_2_2018** was used. These embeddings were trained using texts obtained from sites in the .ru and .рф domains.

Also, I tried to use sentence-level embeddings like [USE](https://tfhub.dev/google/universal-sentence-encoder-multilingual/3) and [RusVectores ELMo](https://rusvectores.org/ru/models/), but the clustering results were either comparable to W2V or worse.

### Vector weights
The vector of each word in `context` was weighted by varying weights (**chi2 coefficients, tf_idf, weights based on word frequencies**). 

This approach [showed](http://www.dialog-21.ru/media/4538/arefyevn_ermolaevp_panchenkoa.pdf) promising results.

## Clustering
Several [algorithms](http://www.dialog-21.ru/media/4385/panchenko.pdf) were used:
* Affinity Propagation;
* Spectral Clustering;
* Agglomerative clustering;
* K-Means;
* HDBSCAN;
* Birch.

Best of all result for the `active-dict` dataset was shown by [combination](http://www.dialog-21.ru/media/4311/kutuzovab.pdf) of **Affinity Propagation** and **Agglomerative clustering**.

**Affinity Propagation** would find the number of clusters for each word and **Agglomerative clustering**, by using this number of clusters, would cluster context embeddings into groups.

For the `active-rutenten` dataset, clustering in one stage using **Agglomerative clustering** algorithm with the predefined number of **3** clusters worked relatively well.

Also, if you don't want to use a predefined number of clusters - **HDBSCAN** has shown relatively good results either.

## Results
[Adjusted Rand index (ARI)](https://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index) was used as an evaluation score.

Results were compared to **[AdaGram model](https://github.com/lopuhin/python-adagram)** provided by the organizers and **randomly generated labels** (from 1 to 3 different labels).

| Dataset/ARI  | AdaGram  | Trivial (random-1-3)  | My approach  |
|---|---|---|---|
| active-dict **train**  |0.159930|0.014134|**0.240510**|
| active-dict **test**  |0.161189|-0.006112|**0.237777**|
| active-rutenten **train** |0.195162|0.000677|**0.216266** (3 clusters)|

## References
1. Panchenko, A., Lopukhina, A., Ustalov, D., Lopukhin, K., Arefyev, N., Leontyev, A., & Loukachevitch, N. (2018). RUSSE'2018: a shared task on word sense induction for the Russian language. arXiv preprint arXiv:1803.05795.
2. Kutuzov A., Kuzmenko E. (2017) WebVectors: A Toolkit for Building Web Interfaces for Vector Semantic Models. In: Ignatov D. et al. (eds) Analysis of Images, Social Networks and Texts. AIST 2016. Communications in Computer and Information Science, vol 661. Springer, Cham
3. Arefyev, N., Ermolaev, P., & Panchenko, A. (2018). How much does a word weigh? Weighting word embeddings for word sense induction. arXiv preprint arXiv:1805.09209.
4. Kutuzov, A. (2018). Russian word sense induction by clustering averaged word embeddings. arXiv preprint arXiv:1805.02258.
5. Bartunov, S., Kondrashkin, D., Osokin, A., & Vetrov, D. (2016, May). Breaking sticks and ambiguities with adaptive skip-gram. In artificial intelligence and statistics (pp. 130-138).
