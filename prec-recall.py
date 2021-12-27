from search_data import search
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)


def partition_every(n, source):
    return [source[i::n] for i in range(n)]


def process_ground_truth():
    with open("./ground-truth-unique.txt") as f:
        contents = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    [queries, classes, filepaths] = partition_every(3, contents)
    return pd.DataFrame({"query": queries, "class": classes, "path": filepaths})


def compute_prec_recall(gt, results):
    precision = recall = 0
    for (i, ground_truth) in gt.iterrows():
        df = results[i]
        filtered = df[(df["name"] == ground_truth["class"]) & (df["file"] == ground_truth["path"])]
        if not len(filtered.index):
            continue
        [position] = filtered.index
        precision += 1 / (position + 1)
        recall += 1
    precision /= len(gt.index)
    recall /= len(gt.index)
    return precision, recall


def main():
    ground_truth = process_ground_truth()
    queries = ground_truth["query"].to_list()
    [all_freq, all_tfidf, all_lsi, all_doc2vec, all_lsi_vec, all_doc2vec_vec] = search(queries)
    tsne = TSNE(n_components=2, perplexity=50, init='pca', n_iter=5000, random_state=420)
    [embedded_vec_lsi, embedded_vec_doc2vec] = [
        [tsne.fit_transform(vector) for vector in vectors]
        for vectors in [all_lsi_vec, all_doc2vec_vec]
    ]

    df_vec_lsi = [[[*y, i] for y in x] for i, x in enumerate(embedded_vec_lsi)]
    df_vec_lsi = [y for x in df_vec_lsi for y in x]
    df_vec_lsi = pd.DataFrame(df_vec_lsi, columns=["x", "y", "idx"])
    plt.figure()
    sns.scatterplot(data=df_vec_lsi, x="x", y="y", hue='idx', palette=sns.color_palette("tab10"))
    plt.show()

    df_vec_doc2vec = [[[*y, i] for y in x] for i, x in enumerate(embedded_vec_doc2vec)]
    df_vec_doc2vec = [y for x in df_vec_doc2vec for y in x]
    df_vec_doc2vec = pd.DataFrame(df_vec_doc2vec, columns=["x", "y", "idx"])
    plt.figure()
    sns.scatterplot(data=df_vec_doc2vec, x="x", y="y", hue='idx', palette=sns.color_palette("tab10"))
    plt.show()

    freq_prec, freq_recall = compute_prec_recall(ground_truth, all_freq)
    print(f"{freq_prec=}, {freq_recall=}")
    tfidf_prec, tfidf_recall = compute_prec_recall(ground_truth, all_tfidf)
    print(f"{tfidf_prec=}, {tfidf_recall=}")
    lsi_prec, lsi_recall = compute_prec_recall(ground_truth, all_lsi)
    print(f"{lsi_prec=}, {lsi_recall=}")
    doc2vec_prec, doc2vec_recall = compute_prec_recall(ground_truth, all_doc2vec)
    print(f"{doc2vec_prec=}, {doc2vec_recall=}")


if __name__ == "__main__":
    main()
