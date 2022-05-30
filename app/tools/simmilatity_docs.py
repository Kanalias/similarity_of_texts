from typing import List

import numpy as np
import pandas

from app.methods.word_2_vec import W2V


class SimilarityDocs:

    def __init__(self, w2v):
        self.score = 0.98753
        self.w2v: W2V = w2v

    def print_similarity(self, df: pandas.DataFrame, source_file, target_files):
        """ {
                    "source_texts": source_texts,
                    "target_texts": target_texts,
                    "sims": sims,
                    "source_index_texts": source_index_texts,
                    "source_filter_texts": source_filter_texts,
                    "source_file": [source_file["file_name"] for i in range(len(sims))],
                    "source_file_index": [source_file["index"] for i in range(len(sims))],
                    "target_index_texts": target_index_texts,
                    "target_filter_texts": target_filter_texts,
                    "target_file": [target_file["file_name"] for i in range(len(sims))],
                    "target_file_index": [target_file["index"] for i in range(len(sims))],
                } """

        for target_file in target_files:
            _df = df.loc[df["target_file_index"] == target_file["index"]]
            n = len(pandas.unique(_df['source_index_texts']))
            all = len(source_file["texts"])
            print("Сходство:", n / all, "SourceFile:", source_file["file_name"], "TargetFile:",
                  target_file["file_name"], sep="\t")

    def similarity(self, source_file: dict, target_files: List):
        """
        source_file: {"file_name": str, "texts": list[str], "filter_texts": list[str] }
        target_files: list[source_file]
        """

        """
            Сравнение документов:
            Попробовать сравнивать различными способами предложения, параграфы.
            Как учитывать сравнение?
            Если сравниваемый элемент похож на другой больше чем score (выбрать какое), то считаем что элемент похож
            Общий процент похожести, поделить количество похожих элементов на количество всех
        """

        source_vectors = [self.w2v.get_avg_vector(source_doc) for source_doc in source_file["filter_texts"]]

        df = pandas.DataFrame()

        for target_file in target_files:
            target_vectors = [self.w2v.get_avg_vector(target_doc) for target_doc in target_file["filter_texts"]]

            sims = np.array([np.array([self.w2v.similarity(source_vector, target_vector)
                                       for target_vector in target_vectors])
                             for source_vector in source_vectors])

            max_elements = np.argwhere(sims >= self.score)

            source_filter_texts = [source_file["filter_texts"][indexs[0]] for indexs in max_elements]
            source_texts = [source_file["texts"][indexs[0]] for indexs in max_elements]
            source_index_texts = [indexs[0] for indexs in max_elements]

            target_filter_texts = [target_file["filter_texts"][indexs[1]] for indexs in max_elements]
            target_texts = [target_file["texts"][indexs[1]] for indexs in max_elements]
            target_index_texts = [indexs[1] for indexs in max_elements]

            sims = [sims[indexs[0]][indexs[1]] for indexs in max_elements]

            _df = pandas.DataFrame(
                data={
                    "source_texts": source_texts,
                    "target_texts": target_texts,
                    "sims": sims,
                    "source_index_texts": source_index_texts,
                    "source_filter_texts": source_filter_texts,
                    "source_file": [source_file["file_name"] for i in range(len(sims))],
                    "source_file_index": [source_file["index"] for i in range(len(sims))],
                    "target_index_texts": target_index_texts,
                    "target_filter_texts": target_filter_texts,
                    "target_file": [target_file["file_name"] for i in range(len(sims))],
                    "target_file_index": [target_file["index"] for i in range(len(sims))],
                }
            )

            df = pandas.concat([df, _df])

        self.print_similarity(df, source_file, target_files)

        # for source_doc in source_file["filter_texts"]:
        #     source_vector = self.w2v.get_avg_vector(source_doc)
        #
        #     sims = [self.w2v.similarity(source_vector, self.w2v.get_avg_vector(target_document))
        #             for target_document in target_file]
        #
        #     source_text = source_texts[index]
        #
        #     _df = pandas.DataFrame(data={
        #         "source_texts": [source_text for i in range(len(sims))],
        #         "source_documents": [source_document for i in range(len(sims))],
        #         "target_texts": target_texts,
        #         "target_documents": target_docs,
        #         "similarity": sims
        #     })
        #
        #     max = _df['similarity'].max()
        #     if max >= self.score:
        #         _df_filter = _df[_df['similarity'] >= max]
        #         df = pandas.concat([df, _df_filter])
        #
        # print("Сходство:", df["similarity"].sum() / df["similarity"].count())
