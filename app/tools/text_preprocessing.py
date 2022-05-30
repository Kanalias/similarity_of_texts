import os
from typing import List, Tuple
import re
import nltk
from gensim.models import doc2vec
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("stopwords")
from nltk.corpus import stopwords
import pymorphy2


class TextPreprocessing:
    __punctuation = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~»«–“”"

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        self.stop_words = list(stopwords.words("russian"))
        self.my_stop_words = self.__read_stop_words()

    @staticmethod
    def __read_stop_words():
        with open(f"{os.path.dirname(os.path.abspath(__file__))}/data/my_stop_words.txt", "r", encoding="utf-8") as f:
            return [item for item in f.read().splitlines()]

    def text_preprocessing(self, text: str,
                           is_lower: bool = True,
                           delete_punctuation: bool = True,
                           delete_stopwords: bool = True,
                           lemmatization: bool = True) -> Tuple[List[str], str]:
        if is_lower:
            text = text.lower()

        text = re.sub(r"https?://[^,\s]+,?", "", text)# Удаление URL

        words = word_tokenize(text, language="russian")

        cleaned_words = []

        for word in words:

            word = re.sub(r"_{2,}", "", word)

            if not word:
                continue

            if delete_punctuation and word in self.__punctuation:
                continue

            if word in self.my_stop_words:
                continue

            if lemmatization:
                word = self.morph.parse(word)[0].normal_form

            if delete_stopwords and word in self.stop_words:
                continue

            cleaned_words.append(word)

        return cleaned_words, text

    def get_filter_documents(self, texts, is_paragraph: bool = True):
        _documents = []
        _documents_orig = []
        for text in texts:

            if is_paragraph:
                filter_text, orig_text = self.text_preprocessing(text)
                if filter_text:
                    _documents.append(filter_text)
                    _documents_orig.append(orig_text)
            else:
                sentences = sent_tokenize(text)

                for sentence in sentences:
                    filter_text, orig_text = self.text_preprocessing(sentence)
                    if filter_text:
                        _documents.append(filter_text)
                        _documents_orig.append(orig_text)

        return _documents, _documents_orig

    # def d2v_text_preprocessing(text: str,
    #                            is_lower: bool = True,
    #                            delete_punctuation: bool = True):
    #     words = text_preprocessing(text, is_lower, delete_punctuation)
    #     return doc2vec.TaggedDocument(words, [i])
