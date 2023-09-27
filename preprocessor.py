import ftfy
import spacy
import os
import json
from tqdm import tqdm
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

    def process_file(self, file_path, landmark_embeddings, progress_bar):
        with open(file_path, "r") as file:
            for line in file:
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
        progress_bar.update(1)

    def process_folder(self, folder, landmark_embeddings, debug, datadir=DATA_PATH):
        """
        Process all files in a folder in a specific directory. Threads are used to speed up the process.
        Every file is processed in a separate thread.
        """
        folder_path = os.path.join(datadir, folder)
        num_files = len(os.listdir(folder_path))

        threads = []  # Store the thread objects

        for file_nr, filename in enumerate(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)

            # Create and start a thread for each file
            progress_bar = tqdm(
                desc=f"Processing '{filename}' in folder '{folder}'",
                position=file_nr,
                total=num_files,
                dynamic_ncols=True,
            )
            thread = threading.Thread(
                target=self.process_file,
                args=(file_path, landmark_embeddings, progress_bar),
            )
            thread.start()
            threads.append(thread)  # Store the thread object

            if debug:
                print(
                    f"{file_nr+1}/{num_files} - Started processing '{filename}' in folder '{folder}'"
                )

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        if debug:
            print(f"Folder {folder} is processed")

        return list(self.shared_page_dictionary)
