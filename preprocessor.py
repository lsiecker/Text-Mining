import warnings
import ftfy
import spacy
from spacy.tokens import Doc, DocBin
import random
import os
import json
from tqdm import tqdm, trange
import multiprocessing
import re


# Set the root directory of the project
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
        with open(file_path, "r", encoding="utf-8") as file:
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

    def process_file_title(self, file_path, title):
        # Load the JSON data
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                info_dict = json.loads(line)
                if (
                    title in info_dict["text"]
                    and info_dict not in self.shared_page_dictionary
                    and info_dict["text"] != ""
                ):
                    self.shared_page_dictionary.append(info_dict)

    def process_folder(
        self, folder, landmark_embeddings, debug, title, nlp, datadir=DATA_PATH
    ):
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

            if nlp:
                self.process_file_nlp(file_path, landmark_embeddings)
            else:
                self.process_file_title(file_path, title)

            if debug:
                print(
                    f"{file_nr+1}/{num_files} - Started processing '{filename}' in folder '{folder}'"
                )
        if debug:
            print(f"Folder {folder} is processed")

        return list(self.shared_page_dictionary)

    def process_export(self, export_data):
        # Create dictionaries to store labels and their relations
        label_data = {}
        relation_data = {}
        training_data = []

        for data in export_data:
            for user in data["annotations"]:
                for item in user["result"]:
                    label_list = []
                    if item["type"] == "labels":
                        label_list.append(
                            (
                                item["value"]["start"],
                                item["value"]["end"],
                                item["value"]["labels"][0],
                            )
                        )
                        label_id = item["id"]
                        label_value = item["value"]["text"]
                        label_data[label_id] = label_value
                    elif item["type"] == "relation":
                        from_id = item["from_id"]
                        to_id = item["to_id"]
                        relation_labels = item["labels"]
                        if from_id in label_data and to_id in label_data:
                            relation_data[
                                (label_data[from_id], label_data[to_id])
                            ] = relation_labels
                    if label_list != []:
                        training_data.append((data["data"]["text"], label_list))

        return training_data, relation_data

    # Check for overlapping entities
    def check_overlap(entities):
        for i, (start1, end1, label1) in enumerate(entities):
            for j, (start2, end2, label2) in enumerate(entities):
                if i != j:
                    if start1 < end2 and start2 < end1:
                        print(
                            f"Overlapping entities: {label1} ({start1}-{end1}) and {label2} ({start2}-{end2})"
                        )

    def preprocess_spacy(self, training_data, split_ratio=0.8, warn=False):
        nlp = spacy.blank("en")

        # Shuffle the training data to ensure randomness
        random.shuffle(training_data)

        # Split the data into training and development sets
        split_index = int(len(training_data) * split_ratio)
        train_data = training_data[:split_index]
        dev_data = training_data[split_index:]

        # Create train.spacy
        train_db = DocBin()
        for text, annotations in train_data:
            doc = nlp(text)
            ents = []
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                if span is None:
                    msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                    if warn == True:
                        warnings.warn(msg)
                else:
                    ents.append(span)
            doc.ents = ents
            train_db.add(doc)

        train_save_path = os.path.join(ROOT_DIR, "data", "train.spacy")
        train_db.to_disk(train_save_path)

        # Create dev.spacy
        dev_db = DocBin()
        for text, annotations in dev_data:
            doc = nlp(text)
            ents = []
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                if span is None:
                    msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                    if warn == True:
                        warnings.warn(msg)
                else:
                    ents.append(span)
            doc.ents = ents
            dev_db.add(doc)

        dev_save_path = os.path.join(ROOT_DIR, "data", "dev.spacy")
        dev_db.to_disk(dev_save_path)
