import sys
import pandas as pd
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel, doc2vec
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

pd.set_option('display.max_colwidth', None)


def read_corpus(lines: [str], tokens_only=False):
    for i, line in enumerate(lines):
        tokens = simple_preprocess(line)
        if tokens_only:
            yield tokens
        else:
            yield doc2vec.TaggedDocument(tokens, [i])


def search(queries: [str]):
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

    tfidf_results = []
    for query in queries:
        query_bow = dictionary.doc2bow(query.split())
        sims = index[tfidf[query_bow]]
        sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        tfidf_data = []
        for i, s in sims[:5]:
            row = data.loc[i][["name", "file", "type", "line"]]
            row["similarity"] = s
            tfidf_data.append(row)
        tfidf_data = pd.concat(tfidf_data, axis=1).T.reset_index(drop=True)
        tfidf_results.append(tfidf_data)

    # Search by LSI
    corpus_tfidf = tfidf[data["corpus_bow"].to_list()]
    lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = lsi[corpus_tfidf]
    index = MatrixSimilarity(corpus_lsi)

    lsi_results = []
    for query in queries:
        query_bow = dictionary.doc2bow(query.split())
        vec_lsi = lsi[tfidf[query_bow]]
        sims = abs(index[vec_lsi])
        sims = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)

        lsi_data = []
        for i, s in sims[:5]:
            row = data.loc[i][["name", "file", "type", "line"]]
            row["similarity"] = s
            lsi_data.append(row)
        lsi_data = pd.concat(lsi_data, axis=1).T.reset_index(drop=True)
        lsi_results.append(lsi_data)

    # Search by doc2vec
    corpus_doc2vec = list(read_corpus(data["processed_doc"].to_list()))
    model = doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
    model.build_vocab(corpus_doc2vec)
    model.train(corpus_doc2vec, total_examples=model.corpus_count, epochs=model.epochs)

    doc2vec_results = []
    for vector in list(read_corpus(queries, tokens_only=True)):
        inferred_vec = model.infer_vector(vector)
        sims = model.docvecs.most_similar([inferred_vec], topn=5)
        doc2vec_data = []
        for i, s in sims[:5]:
            row = data.loc[i][["name", "file", "type", "line"]]
            row["similarity"] = s
            doc2vec_data.append(row)
        doc2vec_data = pd.concat(doc2vec_data, axis=1).T.reset_index(drop=True)
        doc2vec_results.append(doc2vec_data)

    return [tfidf_results, lsi_results, doc2vec_results]


def main():
    [(top5_tfidf, top5_lsi, top5_doc2vec)] = list(zip(*search(["Optimizer that implements the Adadelta algorithm"])))

    print("Top 5 - TFIDF:")
    print(top5_tfidf)

    print("Top 5 - LSI:")
    print(top5_lsi)

    print("Top 5 - Doc2Vec:")
    print(top5_doc2vec)


if __name__ == "__main__":
    main()
