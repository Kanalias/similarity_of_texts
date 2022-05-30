import json
from typing import List, Optional, Union, Generator
import os
import win32com.client as win32
import docx


class DocReader:
    DOC = "doc"
    DOCX = "docx"
    EXTENSIONS = ["docx", "doc"]

    def __file_conversion(self, path: str) -> Optional[str]:
        try:
            word = win32.Dispatch('Word.Application')
            wb = word.Documents.Open(path)
            wb.SaveAs2(path, FileFormat=16)
            wb.Close()
            word.Quit()
            path += "x"
            return path
        except Exception as e:
            print("Не удалось отрыть Microsoft Office Word или преобразовать файл. (файл будет пропущен)\nФайл:", path)
            return None

    def read_file(self, path: str) -> Optional:

        if path.lower().endswith(self.DOC):
            path = self.__file_conversion(path)

        if not path or not path.endswith(self.DOCX):
            return None

        doc = docx.Document(path)

        return doc

    def read_files(self, dirname: str, extensions: List = None) -> Generator:
        if extensions is None:
            extensions = self.EXTENSIONS

        file_names = os.listdir(dirname)
        paths = map(lambda name: os.path.join(dirname, name), file_names)

        extensions_filter = lambda name: name.lower().split(".")[-1] in extensions

        paths = filter(extensions_filter, paths)
        file_names = filter(extensions_filter, file_names)

        for path, file_name in zip(paths, file_names):
            try:
                file = self.read_file(path)
                if file:
                    yield {"file_name": file_name, "file": file}
            except Exception as e:
                print(e)
                continue

    def read_json(self, file_name: str):
        with open(f"data/json/{file_name}", "r") as json_file:
            return json.load(json_file)

    def save_json(self, data: List[dict], file_name: str):
        with open(f"data/json/{file_name}", "w") as f:
            json.dump(data, f)
