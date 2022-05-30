import itertools
from collections import Counter
from typing import List, Union
import numpy as np
from gensim.models import Word2Vec
from gensim import downloader
from scipy import spatial
from multiprocessing import cpu_count
from app.tools import text_preprocessing


class W2V:

    def __init__(self):
        self.model = None
        self.vector_size = 300
        self.keyed_vectors = None
        self.index2word_set = None

    def train(self, documents: List, update: bool = False, compute_loss: bool = False) -> None:

        # model = Word2Vec(window=10, vector_size=self.vector_size, epochs=30, workers=6, min_count=2, negative=15)
        # model = Word2Vec(workers=cpu_count(), vector_size=self.vector_size,
        #                  epochs=40, window=10, min_count=5, negative=5,
        #                  sg=1, ns_exponent=0.75, cbow_mean=1, seed=1, shrink_windows=True
        #                  )
        if update:
            self.model.compute_loss = compute_loss
            self.model.build_vocab(documents, update=True)
            self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
            self.keyed_vectors = self.model.wv
            self.index2word_set = set(self.model.wv.index_to_key)
        else:
            model = Word2Vec(workers=cpu_count(), vector_size=self.vector_size,
                             epochs=40, window=10, min_count=5, negative=5, sample=1e-3,
                             sg=1, ns_exponent=0.75, cbow_mean=1, seed=1, shrink_windows=True,
                             compute_loss=compute_loss
                             )

            model.build_vocab(documents)
            model.train(documents, total_examples=model.corpus_count, epochs=model.epochs, report_delay=1)
            self.keyed_vectors = model.wv
            self.index2word_set = set(model.wv.index_to_key)
            self.model = model

    def save(self, name: str) -> None:
        self.model.save(f"app/models/{name}")

    def load(self, path: str, is_api: bool = False) -> None:
        if is_api:
            self.keyed_vectors = downloader.load(path)
        else:
            self.model = Word2Vec.load(f"app/models/{path}")
            self.keyed_vectors = self.model.wv

    def get_avg_vector(self, document: Union[str, List]):
        if self.index2word_set is None:
            self.index2word_set = set(self.keyed_vectors.index_to_key)

        if isinstance(document, str):
            words, _ = text_preprocessing.text_preprocessing(document)
        else:
            words = document

        feature_vec = np.zeros((self.vector_size,), dtype='float32')
        n_words = 0
        for word in words:
            if word in self.index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, self.keyed_vectors[word])

        if n_words > 0:
            feature_vec = np.divide(feature_vec, n_words)

        return feature_vec

    def get_sum_vector(self, document: Union[str, List]):
        a = np.array(
            [self.keyed_vectors[token] for token in document.split() if token in self.keyed_vectors.index_to_key])
        return np.sum(a, axis=0)

    def map_word_frequency(self, document):
        return Counter(itertools.chain(*document))

    def get_sif_vectors(self, sentence1, sentence2):
        sentence1 = [token for token in sentence1.split() if token in self.keyed_vectors.index_to_key]
        sentence2 = [token for token in sentence2.split() if token in self.keyed_vectors.index_to_key]
        word_counts = self.map_word_frequency((sentence1 + sentence2))
        embedding_size = self.vector_size  # size of vectore in word embeddings
        a = 0.001
        sentence_set = []
        for sentence in [sentence1, sentence2]:
            vs = np.zeros(embedding_size)
            sentence_length = len(sentence)
            for word in sentence:
                a_value = a / (a + word_counts[word])  # smooth inverse frequency, SIF
                vs = np.add(vs, np.multiply(a_value, self.keyed_vectors[word]))  # vs += sif * word_vector
            vs = np.divide(vs, sentence_length)  # weighted average
            sentence_set.append(vs)
        return sentence_set

    # def vectorize(self, doc: str) -> np.ndarray:
    #     doc = doc.lower()
    #     words = [w for w in doc.split()]
    #     word_vecs = []
    #     for word in words:
    #         try:
    #             vec = self.model.wv[word]
    #             word_vecs.append(vec)
    #         except KeyError:
    #             # Ignore, if the word doesn't exist in the vocabulary
    #             pass
    #
    #     # Assuming that document vector is the mean of all the word vectors
    #     # PS: There are other & better ways to do it.
    #     vector = np.mean(word_vecs, axis=0)
    #     return vector

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray):
        return 1 - spatial.distance.cosine(vec1, vec2) \
            if np.count_nonzero(vec1 == 0) != self.vector_size and np.count_nonzero(vec2 == 0) != self.vector_size \
            else 0
