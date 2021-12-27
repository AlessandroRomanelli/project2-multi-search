import re
from collections import defaultdict

import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.parsing.preprocessing import preprocess_string, lower_to_unicode, remove_stopwords, strip_punctuation, \
    strip_numeric, strip_non_alphanum, split_alphanum, stem_text, strip_short, strip_multiple_whitespaces, strip_tags
from gensim.similarities import SparseMatrixSimilarity
from gensim.utils import simple_preprocess

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

def split_words(s: str):
    return " ".join([" ".join(split_camelcase(x)) if is_camelcase(x) else x for x in s.split()])


def split_camelcase(s: str):
    return re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', s)


def is_camelcase(s: str):
    return len(split_camelcase(s)) > 0


def preprocess_queries(s: str):
    return " ".join(preprocess_string(split_words(s), filters=[
        lower_to_unicode,
        split_alphanum,
        strip_short,
        strip_numeric
    ]))


def preprocess_document(s: str):
    return " ".join(preprocess_string(split_words(s), filters=[
        lower_to_unicode,
        split_alphanum,
        strip_short,
        strip_numeric
    ]))


def preprocess_document_train(s: str):
    return " ".join(preprocess_string(split_words(s), filters=[
        lower_to_unicode,
        split_alphanum,
        strip_tags,
        strip_punctuation,
        strip_multiple_whitespaces,
        remove_stopwords,
        strip_short,
        strip_non_alphanum,
        stem_text
    ]))


def read_results():
    return pd.read_csv("./results.csv").fillna("")


def get_documents(training=False):
    df = read_results()
    preprocess_fn = preprocess_document if not training else preprocess_document_train
    documents = [preprocess_fn(doc) for doc in (df["name"] + " " + df["comment"]).to_list()]
    frequency = defaultdict(int)
    for doc in documents:
        for word in doc.split():
            frequency[word] += 1

    documents = [
        " ".join([word for word in doc.split() if frequency[word] > 1 and word not in ["main", "test"]]) for
        doc in documents]
    return [x for x in enumerate(documents) if len(x[1])]


def search(queries: [str]):
    queries = [preprocess_queries(x) for x in queries]
    results = read_results()
    documents = get_documents()
    documents_train = get_documents(True)
    dictionary = Dictionary([doc.split() for i, doc in documents])
    dictionary_train = Dictionary([doc.split() for i, doc in documents_train])
    corpus = [dictionary.doc2bow(doc.split()) for i, doc in documents]
    corpus_train = [dictionary_train.doc2bow(doc.split()) for i, doc in documents_train]

    def get_top5(sims, docs):
        top5 = []
        for i, similarity in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:5]:
            idx, doc = docs[i]
            result = results.iloc[idx].copy()
            result["similarity"] = similarity
            top5.append(result)
        return pd.DataFrame(top5).reset_index(drop=True)

    index = SparseMatrixSimilarity(corpus, num_features=len(dictionary))

    def search_by_freq(query: str):
        query_bow = dictionary.doc2bow(query.split())
        return get_top5(index[query_bow], documents)

    tfidf = TfidfModel(corpus, normalize=True)
    tfidf_index = SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary))

    def search_by_tfidf(query: str):
        query_bow = dictionary.doc2bow(query.split())
        return get_top5(tfidf_index[query_bow], documents)

    lsi_tfidf = TfidfModel(corpus_train, normalize=True)
    lsi_tfidf_corpus = lsi_tfidf[corpus_train]
    lsi = LsiModel(lsi_tfidf_corpus, id2word=dictionary_train, num_topics=300)
    lsi_corpus = lsi[lsi_tfidf_corpus]
    lsi_index = SparseMatrixSimilarity(lsi_corpus, num_features=len(dictionary_train))

    def search_by_lsi(query: str):
        query = preprocess_document_train(query)
        vec_bow = dictionary_train.doc2bow(query.split())
        vec_lsi = lsi[vec_bow]
        return get_top5(lsi_index[vec_lsi], documents_train)

    def vector_by_lsi(query: str):
        top5_vec = []
        # query = preprocess_document_train(query)
        vec_bow = dictionary_train.doc2bow(query.split())
        vec_lsi = lsi[vec_bow]
        sims = lsi_index[vec_lsi]
        for i, similarity in sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:5]:
            top5_vec.append([x[1] for x in lsi_corpus[i]])
        return top5_vec

    train_corpus = [TaggedDocument(doc.split(), [i]) for i, doc in documents_train]
    doc2vec = Doc2Vec(train_corpus, vector_size=300, dm=0, window=10, min_count=2, epochs=60, seed=420)

    def search_by_doc2vec(query: str):
        query = preprocess_document_train(query)
        vec = doc2vec.infer_vector(query.split())
        sims = doc2vec.dv.most_similar([vec], topn=5)
        top5 = []
        for i, similarity in sims:
            result = results.iloc[i].copy()
            result["similarity"] = similarity
            top5.append(result)
        return pd.DataFrame(top5).reset_index(drop=True)

    def vector_by_doc2vec(query: str):
        top5_vec = []
        query = preprocess_document_train(query)
        vec = doc2vec.infer_vector(query.split())
        sims = doc2vec.dv.most_similar([vec], topn=5)
        for i, s in sims:
            top5_vec.append(doc2vec.dv[i])
        return top5_vec

    [freq_top5, tfidf_top5, lsi_top5, doc2vec_top5, lsi_top5_vecs, doc2vec_top5_vecs] = list(zip(*[
        [
            search_by_freq(query),
            search_by_tfidf(query),
            search_by_lsi(query),
            search_by_doc2vec(query),
            vector_by_lsi(query),
            vector_by_doc2vec(query)
        ] for query in queries]))

    return [freq_top5, tfidf_top5, lsi_top5, doc2vec_top5, lsi_top5_vecs, doc2vec_top5_vecs]


if __name__ == "__main__":
    [[d1], [d2], [d3], [d4], [v5], [v6]] = search(["Optimizer that implements the Adadelta algorithm"])
    print(d1[["name", "file", "similarity"]].to_latex())
    print(d2[["name", "file", "similarity"]].to_latex())
    print(d3[["name", "file", "similarity"]].to_latex())
    print(d4[["name", "file", "similarity"]].to_latex())
