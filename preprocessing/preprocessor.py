import ftfy
import pandas as pd
import spacy
import os
import json
from tqdm import tqdm
import multiprocessing
import re
import nltk

# Download the necessary NLTK data for sentence splitting
nltk.download("punkt", quiet=True)

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname("preprocessing.ipynb"), os.pardir)
)
DATA_PATH = os.path.join(ROOT_DIR, "data")


class Preprocessor:
    def __init__(self, basedir: str = ROOT_DIR, spacy_lib: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_lib)
        self.basedir = basedir
        self.manager = multiprocessing.Manager()
        self.shared_page_dictionary = self.manager.list()

    def fix_unicode(self, data: list):
        """
        Cleans UNIX characters from a given article text and stores it in a new value stored under the 'text' key

        :param data: List of dictionaries where each dictionary represents
            a Wikipedia article and its metadata
        :return: A list of dictionaries where again
            each dictionary represents a Wikipedia article, but now with
            an additional key-value pair with the cleaned article text.
        """
        output = []
        for article in tqdm(data):
            text = article["text"]
            article["original_text"] = text
            unicode_fix = ftfy.fix_text(text)
            article["text"] = unicode_fix
            output.append(article)
        return output

    def writeFile(self, data: list, name: str, basedir: str = ROOT_DIR):
        """
        Saves a dataset to a JSON file for a given name.

        :param data: List of dictionaries where each dictionary represents
            a Wikipedia article and its metadata
        :param name: Name of the stored file
        :param basedir: Path of the base directory
        :return: None

        """
        file_path = os.path.join(basedir, "data\\", name)
        with open(file_path, "w") as file:
            json.dump(data, file, indent=2)

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

    def save_file(self, data: list, folder: str):
        """
        Saves a dataset to multiple JSON files in a given folder.

        :param data: List of dictionaries where each dictionary represents
            a Wikipedia article and its metadata
        :param folder: Name of the folder where the files are stored
        :return: None
        """
        for i, article in enumerate(data):
            title = (
                "".join(char for char in article["title"].lower() if char.isalnum())
                + ".json"
            )
            file_path = os.path.join(self.basedir, "data\\", folder, title)
            text = re.sub(r"[^\x00-\x7F]+", "", article["text"])
            to_dump = {
                "id": i,
                "data": {
                    "title": article["title"],
                    "text": text,
                },
            }
            with open(file_path, "w") as file:
                # Save article text to file
                json.dump(to_dump, file)

    def clean_alphanumeric(self, data: list, pattern=r"\W+"):
        """
        Cleans non-alphanumeric characters from a given article text and stores it in a new value stored under the 'text' key

        :param data: List of dictionaries where each dictionary represents
            a Wikipedia article and its metadata
        :param pattern: Regular expression pattern to split the text
        :return: A list of dictionaries where again
            each dictionary represents a Wikipedia article, but now with
            an additional key-value pair with the cleaned article text.
        """
        cleaned_articles = []

        # Loop through each text in the list
        for article in tqdm(data):
            text = article["text"]
            # Tokenize the text using the regular expression pattern
            words = re.split(pattern, text)
            # Count occurrences of non-alphanumeric words
            char_words = [word for word in words if word.isalnum()]
            article["text"] = " ".join(char_words)
            cleaned_articles.append(article)
        return cleaned_articles

    def clean_html(self, dataset: pd.DataFrame, column: str):
        """
        Cleans html tags from a given column in a dataset.

        :param dataset: The dataset that needs to be cleaned
        :param column: The column that needs to be cleaned
        :return: The cleaned dataset
        """
        for i, text in enumerate(tqdm(dataset[column])):
            text = re.sub(r"<.*?>", "", text)
            text = ftfy.fix_text(text)
            dataset.loc[i, column] = text

        return dataset

    def ner_spacy(self, text: str):
        """
        Process a text with the spacy nlp model.

        :param text: The text to be processed
        :return: A document object
        """
        doc = self.nlp(text)
        return doc

    def process_file_nlp(
        self,
        file_path: str,
        landmark_embeddings: list,
        similarity_threshold: float = 0.97,
    ):
        """
        Process a file with the spacy nlp model. And check if the titles of the articles are similar to the landmark embeddings.

        :param file_path: The path to the file that needs to be processed
        :param landmark_embeddings: A list of the landmark embeddings
        :return: A list of the significantly similar pages
        """
        shared_page_dictionary = self.manager.list()
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            for info_dict in tqdm(data, total=len(data)):
                # Handle every seperate wikipedia page
                # Check if the title and the text are not empty strings
                if (
                    len(info_dict.get("title", "")) > 1
                    and len(info_dict.get("text", "")) > 1
                ):
                    title_embedding = self.nlp(info_dict["title"])

                    # Go over all the landmark embeddings and check for similarity
                    for landmark in landmark_embeddings:
                        similarity_score = title_embedding.similarity(landmark)

                        if (
                            similarity_score > similarity_threshold
                            and info_dict not in shared_page_dictionary
                        ):
                            shared_page_dictionary.append(info_dict)
                            break
        return list(shared_page_dictionary)

    def process_file_regex(
        self, file_path: str, title_based: bool, title: str, landmarks: list
    ):
        """
        Process a file and either check if the titles of the articles occur in the landmark list
            or check if the given title occurs in the article.

        :param file_path: The path to the file that needs to be processed
        :param title_based: A boolean that indicates if the title_based method is used
        :param title: The title of the landmark
        :param landmarks: A list of the landmark names
        :return: A list of the articles that were seen as relevant
        """
        # Load the JSON data
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                info_dict = json.loads(line)
                if title_based:
                    if (
                        title in info_dict["text"]
                        and info_dict not in self.shared_page_dictionary
                        and info_dict["text"] != ""
                    ):
                        self.shared_page_dictionary.append(info_dict)
                else:
                    for landmark in landmarks:
                        if (
                            info_dict["title"] in landmark
                            and info_dict not in self.shared_page_dictionary
                            and info_dict["text"] != ""
                        ):
                            self.shared_page_dictionary.append(info_dict)
                            break

    def process_folders(
        self,
        folders: list,
        debug: bool,
        title: str,
        title_based: bool,
        landmarks: list,
        datadir: str = DATA_PATH,
    ):
        """
        Process all files in a folder in a specific directory. Threads are used to speed up the process.
        Every file is processed in a separate thread.

        :param folders: The folders that needs to be processed
        :param debug: A boolean that indicates if the debug mode is on
        :param title: The title of the landmark
        :param title_based: A boolean that indicates if the title_based method is used
        :param landmarks: A list of the landmark embeddings
        :param datadir: The directory where the data is stored
        :return: A list of the shared pages
        """
        for folder in folders:
            folder_path = os.path.join(datadir, folder)
            num_files = len(os.listdir(folder_path))

            for file_nr, filename in enumerate(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, filename)

                self.process_file_regex(file_path, title_based, title, landmarks)

                if debug:
                    print(
                        f"{file_nr+1}/{num_files} - Started processing '{filename}' in folder '{folder}'"
                    )
            if debug:
                print(f"Folder {folder} is processed")

        return list(self.shared_page_dictionary)
