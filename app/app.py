from app.methods.word_2_vec import W2V
from app.tools.doc_reader import DocReader
from app.tools.docx_tools import DocExecuter
from app.tools.simmilatity_docs import SimilarityDocs
from app.tools.text_preprocessing import TextPreprocessing
from typing import List

class App:

    def __init__(self):
        self.w2v = W2V()
        self.text_preprocessing = TextPreprocessing()
        self.doc_reader = DocReader()
        self.doc_executer = DocExecuter()
        self.similarity_docs = SimilarityDocs(self.w2v)

    def read_data(self, dir, save_json_file: str = "data.json"):
        files = []
        for index, doc in enumerate(self.doc_reader.read_files(dir)):
            texts = self.doc_executer.execute_text(doc["file"], split_cell=False, split_paragraphs=True)
            filter_texts = self.text_preprocessing.get_filter_documents(texts, is_sentences=True)[0]

            files.append({
                "index": index,
                "file_name": doc["file_name"],
                "texts": texts,
                "filter_texts": filter_texts
            })

        self.doc_reader.save_json(files, save_json_file)
        return files

    def read_json(self, path: str = "data.json"):
        return self.doc_reader.read_json(path)

    def train(self, files: List[dict], model_name: str):
        documents = [filter_text for file in files for filter_text in file["filter_texts"]]
        self.w2v.train(documents, model_name=model_name, save=True)

    def run(self, is_read_data: bool = False, is_train: bool = False):
        dir = "data\original"
        model_name = "word2vec1.model"

        files = self.read_data(dir, "sentences_data.json") if is_read_data else self.read_json(path="data_new1.json")
        self.train(files=files, model_name=model_name) if is_train else self.w2v.load(model_name)

        self.similarity_docs.similarity(source_files=files[1:2], target_files=files[1:2])
