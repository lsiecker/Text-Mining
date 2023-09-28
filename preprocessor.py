import ftfy
import spacy
import os
import json
from tqdm import tqdm, trange
import multiprocessing
import threading

import re

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

ROOT_DIR = os.path.dirname(
    os.path.dirname("preprocessing.ipynb")
)  # This file is the root of the project
DATA_PATH = os.path.join(ROOT_DIR, "data")


class Preprocessor:
    def __init__(self, basedir, spacy_lib="en_core_web_sm"):
        self.nlp = spacy.load(spacy_lib)
        self.basedir = basedir
        self.manager = multiprocessing.Manager()
        self.shared_page_dictionary = self.manager.list()

    def fix_unicode(self, data):
        print("Cleaning data with ftfy...")
        output = []
        for article in tqdm(data):
            text = article["text"]
            article["original_text"] = text
            unicode_fix = ftfy.fix_text(text)
            article["text"] = unicode_fix
            output.append(article)
        return output

    def writeFile(self, data, name, basedir=ROOT_DIR):
        file_path = os.path.join(basedir, "data\\", name)
        with open(file_path, "w") as file:
            json.dump(data, file, indent=2)

    def loadFile(self, name, basedir=ROOT_DIR):
        file_path = os.path.join(basedir, "data\\", name)
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    def save_file(self, data, folder):
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

    def clean_alphanumeric(self, data, pattern=r"\W+"):
        print("Cleaning data with given regex...")
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

    def ner_spacy(self, text):
        doc = self.nlp(text)
        return doc

    def ner_nltk(self, text):
        return ne_chunk(pos_tag(word_tokenize(text)))

    def process_file_nlp(self, file_path, landmark_embeddings):
        """
        Process a single file. This function is used by the process_folder function.

        :param file_path: The path to the file that needs to be processed
        :param landmark_embeddings: A list of the landmark embeddings
        :return: A list of the shared pages
        """
        print(f"Processing file {file_path}")
        with open(file_path, "r") as file:
            for line in tqdm(file, total=sum(1 for line in open(file_path, "r"))):
                info_dict = json.loads(line)

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
                            similarity_score > 0.97
                            and info_dict not in self.shared_page_dictionary
                        ):
                            self.shared_page_dictionary.append(info_dict)
                            break
                        
    def process_file_title(self, file_path):
        title = "UNESCO World Heritage Site "

        # Load the JSON data
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                info_dict = json.loads(line)
                if title in info_dict['text'] and info_dict not in self.shared_page_dictionary and info_dict['text'] != "":
                            self.shared_page_dictionary.append(info_dict)

    def process_folder(self, folder, landmark_embeddings, debug, datadir=DATA_PATH):
        """
        Process all files in a folder in a specific directory. Threads are used to speed up the process.
        Every file is processed in a separate thread.

        :param folder: The folder that needs to be processed
        :param landmark_embeddings: A list of the landmark embeddings
        :param debug: A boolean that indicates if the debug mode is on
        :param datadir: The directory where the data is stored
        :return: A list of the shared pages
        """
        folder_path = os.path.join(datadir, folder)
        num_files = len(os.listdir(folder_path))

        for file_nr, filename in enumerate(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)

            self.process_file_nlp(file_path, landmark_embeddings)

            if debug:
                print(
                    f"{file_nr+1}/{num_files} - Started processing '{filename}' in folder '{folder}'"
                )
        if debug:
            print(f"Folder {folder} is processed")

        return list(self.shared_page_dictionary)
