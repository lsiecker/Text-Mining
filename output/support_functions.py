import json
import os

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname("preprocessing.ipynb"), os.pardir)
)
DATA_PATH = os.path.join(ROOT_DIR, "data")


class Preprocessor:
    def loadFile(self, name: str, basedir: str = ROOT_DIR):
        """
        Loads a dataset from a JSON file for a given name.

        :param name: Name of the stored file
        :param basedir: Path of the base directory
        :return: List of dictionaries where each dictionary represents
            a Wikipedia article and its metadata
        """
        file_path = os.path.join(basedir, "data\\", name)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
