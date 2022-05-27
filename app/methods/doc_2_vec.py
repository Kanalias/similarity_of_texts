import collections
from typing import List
from gensim.models import Word2Vec, Doc2Vec
from scipy import spatial


class D2V:

    def __init__(self):
        self.model = None
        self.vector_size = 300
        self.index2word_set = None

    def train(self, documents: List):
        model = Doc2Vec(vector_size=self.vector_size, min_count=2, epochs=40, workers=6)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=model.epochs, report_delay=1)
        self.model = model

    def get_vector(self, sentence: str):
        return self.model.infer_vector(sentence.split())

    def test(self, documents):
        ranks = []
        second_ranks = []
        for doc_id in range(len(documents)):
            inferred_vector = self.model.infer_vector(documents[doc_id].words)
            sims = self.model.dv.most_similar([inferred_vector], topn=len(self.model.dv))
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)

            second_ranks.append(sims[1])

            # print('Document ({}): «{}»\n'.format(doc_id, ' '.join(documents[doc_id].words)))
            # print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % self.model)
            # for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2), ('LEAST', len(sims) - 1)]:
            #     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(documents[sims[index][0]].words)))

        counter = collections.Counter(ranks)
        print(counter)

    def similarity(self, vec1, vec2):
        return 1 - spatial.distance.cosine(vec1, vec2)
