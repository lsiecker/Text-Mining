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
import nltk
from nltk.tokenize import sent_tokenize

# Download the necessary NLTK data for sentence splitting
nltk.download("punkt")


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
        return list(self.shared_page_dictionary)

    def process_file_regex(self, file_path, title_based, title, landmarks):
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
        self, folders, debug, title, title_based, landmarks, datadir=DATA_PATH
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

            return self.shared_page_dictionary
        
    def clear_dictionary(self):
        self.shared_page_dictionary.clear()

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

    def process_export_sentences(self, export_data):
        # Create dictionaries to store labels and their relations
        label_data = {}
        relation_data = {}
        training_data = []

        for data in export_data:
            text = data["data"]["text"]
            sentences = sent_tokenize(text)  # Split the text into sentences

            # Initialize a list to store label_list for each sentence
            sentence_label_lists = [[] for _ in sentences]

            for user in data["annotations"]:
                for item in user["result"]:
                    if item["type"] == "labels":
                        label_list = [
                            item["value"]["start"],
                            item["value"]["end"],
                            item["value"]["labels"][0],
                        ]
                        label_id = item["id"]
                        label_value = item["value"]["text"]
                        label_data[label_id] = label_value
                        
                        if label_list != []:
                            # Calculate sentence-level label locations
                            for i, sentence in enumerate(sentences):
                                sentence_start = text.find(sentence)
                                sentence_end = sentence_start + len(sentence)
                                sentence_label_list = []
                                if (
                                    sentence_start <= label_list[0] < sentence_end
                                    and sentence_start < label_list[1] <= sentence_end
                                ):
                                    sentence_label_list = [
                                        [
                                            label_list[0] - sentence_start,
                                            label_list[1] - sentence_start,
                                            label_list[2],
                                        ]
                                    ]
                                sentence_label_lists[i].extend(sentence_label_list)
                            
                    elif item["type"] == "relation":
                        from_id = item["from_id"]
                        to_id = item["to_id"]
                        relation_labels = item["labels"]
                        if from_id in label_data and to_id in label_data:
                            relation_data[
                                (label_data[from_id], label_data[to_id])
                            ] = relation_labels

            # Combine each sentence with its corresponding label_list
            for sent, sent_label_list in zip(sentences, sentence_label_lists):
                if sent_label_list != []:
                    training_data.append([sent, {"entities": sent_label_list}])

        return training_data, relation_data

    def preprocess_json(self, training_data, split_ratio=0.8):
        # Split the data into training and development sets
        split_index = int(len(training_data) * split_ratio)
        train_data = training_data[:split_index]
        dev_data = training_data[split_index:]

        train_save_path = os.path.join(ROOT_DIR, "spacy/assets", "train.json")
        with open(train_save_path, "w") as file:
            # Save article text to file
            json.dump(train_data, file)

        dev_save_path = os.path.join(ROOT_DIR, "spacy/assets", "dev.json")
        with open(dev_save_path, "w") as file:
            # Save article text to file
            json.dump(dev_data, file)

    def preprocess_spacy(self, training_data, split_ratio=0.8, warn=False):
        nlp = spacy.blank("en")

        # Shuffle the training data to ensure randomness
        # random.shuffle(training_data)

        # Split the data into training and development sets
        split_index = int(len(training_data) * split_ratio)
        train_data = training_data[:split_index]
        dev_data = training_data[split_index:]

        # Create train.spacy
        train_db = DocBin()
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            ents = []
            for start, end, label in annotations:
                span = doc.char_span(start, end, label=label)
                if span is None:
                    msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                    if warn == True:
                        warnings.warn(msg)
                else:
                    ents.append(span)
            try:
                doc.ents = ents
            except:
                print(
                    f"Unable to set doc ents, since there is overlap of {ents} in existing doc.ents"
                )
            train_db.add(doc)

        train_save_path = os.path.join(ROOT_DIR, "spacy/corpus", "train.spacy")
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
            try:
                doc.ents = ents
            except:
                print(
                    f"Unable to set doc ents, since there is overlap of {ents} in existing doc.ents"
                )
            dev_db.add(doc)

        dev_save_path = os.path.join(ROOT_DIR, "spacy/corpus", "dev.spacy")
        dev_db.to_disk(dev_save_path)
