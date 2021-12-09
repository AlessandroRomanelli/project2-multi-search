from search_data import search
import pandas as pd


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
        [position] = filtered.index + 1
        stats["precision"] = stats["precision"] + (1 / position)
        stats["recall"] = stats["recall"] + 1
    stats["precision"] = stats["precision"] / len(gt.index)
    stats["recall"] = stats["recall"] / len(gt.index)
    return stats

def main():
    ground_truth = process_ground_truth()
    [all_tfidf, all_lsi, all_doc2vec] = search(ground_truth["query"].to_list())

    tfidf = compute_prec_recall(ground_truth, all_tfidf)
    print(f"{tfidf=}")
    lsi = compute_prec_recall(ground_truth, all_lsi)
    print(f"{lsi=}")
    doc2vec = compute_prec_recall(ground_truth, all_doc2vec)
    print(f"{doc2vec=}")


if __name__ == "__main__":
    main()
