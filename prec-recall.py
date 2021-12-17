from search_data import search, get_top_vectors
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action='ignore',category=FutureWarning)

def partition_every(n, source):
    return [source[i::n] for i in range(n)]


def process_ground_truth():
    with open("./ground-truth-unique.txt") as f:
        contents = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
    [queries, classes, filepaths] = partition_every(3, contents)
    return pd.DataFrame({"query": queries, "class": classes, "path": filepaths})


def compute_prec_recall(gt, results):
    stats = {
        "precision": 0,
        "recall": 0
    }
    for (i,gt) in gt.iterrows():
        df = results[i]
        filtered = df[df["name"] == gt["class"]]
        if not len(filtered.index):
            continue
        [position] = filtered.index
        position += 1
        stats["precision"] = stats["precision"] + (1 / position)
        stats["recall"] = stats["recall"] + 1
    stats["precision"] = stats["precision"] / len(gt.index)
    stats["recall"] = stats["recall"] / len(gt.index)
    return stats


def main():
    ground_truth = process_ground_truth()
    queries = ground_truth["query"].to_list()
    [all_freq, all_tfidf, all_lsi, all_doc2vec] = search(queries)
    [embedded_vec_lsi, embedded_vec_doc2vec] = [
        [TSNE(n_components=2, perplexity=2, n_iter=3000).fit_transform(x) for x in vectors]
        for vectors in get_top_vectors(queries)
    ]

    df_vec_lsi = [[[*y, i] for y in x] for i,x in enumerate(embedded_vec_lsi)]
    df_vec_lsi = [y for x in df_vec_lsi for y in x]
    df_vec_lsi = pd.DataFrame(df_vec_lsi, columns=["x", "y", "idx"])
    plt.figure()
    sns.scatterplot(data=df_vec_lsi, x="x", y="y", hue='idx', palette=sns.color_palette("tab10"))
    plt.show()

    df_vec_doc2vec = [[[*y, i] for y in x] for i,x in enumerate(embedded_vec_doc2vec)]
    df_vec_doc2vec = [y for x in df_vec_doc2vec for y in x]
    df_vec_doc2vec = pd.DataFrame(df_vec_doc2vec, columns=["x", "y", "idx"])
    plt.figure()
    sns.scatterplot(data=df_vec_doc2vec, x="x", y="y", hue='idx', palette=sns.color_palette("tab10"))
    plt.show()

    freq = compute_prec_recall(ground_truth, all_freq)
    print(f"{freq=}")
    tfidf = compute_prec_recall(ground_truth, all_tfidf)
    print(f"{tfidf=}")
    lsi = compute_prec_recall(ground_truth, all_lsi)
    print(f"{lsi=}")
    doc2vec = compute_prec_recall(ground_truth, all_doc2vec)
    print(f"{doc2vec=}")


if __name__ == "__main__":
    main()
