import os
import sys
import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
import re
from collections import defaultdict

stopwords = ["main", "test", "tests", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
             "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
             "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this",
             "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
             "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
             "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
             "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
             "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
             "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own",
             "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


def process_input():
    if len(sys.argv) < 2:
        raise RuntimeError("Must provide a query string as first argument")
    return " ".join([x for x in re.sub("\W+", " ", sys.argv[1].lower()).split() if x not in stopwords])


def main():
    query = process_input()
    print(query)
    # Read previous results
    data = pd.read_csv("results.csv", dtype={0: str, 1: str, 2: int, 3: str, 4: str}).fillna("")

    # Create a document from the union of method name and method comment
    data["document"] = data["name"] + " " + data["comment"]
    # Remove all stopwords from each document
    data["document"] = data["document"].apply(lambda x: " ".join(
        [x for x in re.split("\W|_+", x.strip().lower()) if x not in stopwords]
    ))

    # Count the frequencies of terms in the corpus
    frequency = defaultdict(int)
    for (i, document) in data["document"].iteritems():
        words = [x.strip() for x in document.split(" ") if len(x)]
        for word in words:
            frequency[word] += 1

    # Process the document by removing all terms that appear less than twice in the corpus
    data["processed_doc"] = data["document"].apply(lambda x: " ".join([y for y in x.split() if frequency[y] > 1]))
    data["processed_doc"] = data["processed_doc"].replace("", float("NaN"))

    # Drop all rows with empty document
    data = data.dropna(subset=["processed_doc"]).reset_index(drop=True)

    # Create bag of words
    dictionary = Dictionary([x.split() for x in data["processed_doc"].to_list()])
    data["corpus_bow"] = data["processed_doc"].apply(lambda x: dictionary.doc2bow(x.split()))

    # Search by TF-IDF
    tfidf = TfidfModel(data["corpus_bow"].to_list())
    index = SparseMatrixSimilarity(tfidf[data["corpus_bow"].to_list()], num_features=len(dictionary))
    query_bow = dictionary.doc2bow(query.split())
    sims = index[tfidf[query_bow]]
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
    print("\n\nSimilarity results for TF-IDF:")
    for i, s in sims:
        print(f"Similarity: {s}")
        print(data.loc[i][["name", "file", "type", "line"]].to_string())

    # Search by LSI
    corpus_tfidf = tfidf[data["corpus_bow"].to_list()]
    lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=500)
    corpus_lsi = lsi[corpus_tfidf]

    vec_lsi = lsi[tfidf[query_bow]]
    index = MatrixSimilarity(corpus_lsi)
    sims = abs(index[vec_lsi])
    sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

    print("\n\nSimilarity results for LSI:")
    for i, s in sims[:5]:
        print(f"Similarity: {s}")
        print(data.loc[i][["name", "file", "type", "line"]].to_string())

if __name__ == "__main__":
    main()
