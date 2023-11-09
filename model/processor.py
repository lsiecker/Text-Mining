import warnings
import ftfy
import pandas as pd
import spacy
from spacy.tokens import Doc, DocBin
import os
import json
from tqdm import tqdm
import multiprocessing
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download the necessary NLTK data for sentence splitting
nltk.download("punkt", quiet=True)

# Set the root directory of the project
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname("model_training.ipynb"), os.pardir)
)  # This file is the root of the project
print(f"Root directory: {ROOT_DIR}")
DATA_PATH = os.path.join(ROOT_DIR, "data")


class Processor:
    def __init__(self, basedir: str = ROOT_DIR, spacy_lib: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_lib)
        self.basedir = basedir
        self.manager = multiprocessing.Manager()
        self.shared_page_dictionary = self.manager.list()

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

    def char_to_token(self, text, start, end):
        """
        Converts character-level annotation to token-level annotation.

        :param text: The text that is annotated
        :param start: The start index of the annotation
        :param end: The end index of the annotation
        :return: The start and end index of the annotation in tokens
        """

        doc = self.nlp(text)
        char_span = doc.char_span(start, end, alignment_mode="contract")
        if char_span is None:
            return None, None
        else:
            return char_span.start, char_span.end - 1

    def process_export_sentences(self, export_data: list, ground_truth: bool = False):
        """
        Processes a Label studio export dataset and converts it to a training dataset and relational dataset.

        :param export_data: The Label studio export dataset with all annotations
        :return: A list of the training data and a dictionary of the relations
        """
        # Create dictionaries to store labels and their relations
        label_data = {}
        relational_label_list = []
        relation_data = []
        training_data = []

        entity_info = {}

        for data in export_data:
            text = data["data"]["text"]
            sentences = sent_tokenize(text)  # Split the text into sentences

            # Initialize a list to store label_list for each sentence
            sentence_label_lists = [[] for _ in sentences]
            total_label_list = []
            spans = []

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

                        token_start, token_end = self.char_to_token(
                            text, item["value"]["start"], item["value"]["end"]
                        )

                        if (
                            label_value is not None
                            or item["value"]["labels"] != []
                            or item["value"]["start"] is not None
                            or item["value"]["end"] is not None
                        ):
                            spans.append(
                                {
                                    "text": label_value,
                                    "start": item["value"]["start"],
                                    "end": item["value"]["end"],
                                    "token_start": token_start,
                                    "token_end": token_end,
                                    "type": "span",
                                    "label": item["value"]["labels"][0],
                                }
                            )

                        entity_info[item["id"]] = {
                            "start": item["value"]["start"],
                            "end": item["value"]["end"],
                            "token_start": token_start,
                            "token_end": token_end,
                            "label": item["value"]["labels"][0],
                        }

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
                                    # Check if the span is already in the sentence_label_list, if so take the longest span
                                    if sentence_label_lists[i] != []:
                                        for j, label in enumerate(
                                            sentence_label_lists[i]
                                        ):
                                            if (
                                                label[0] >= label_list[0]
                                                and label[1] <= label_list[1]
                                            ):
                                                sentence_label_lists[i].pop(j)
                                                sentence_label_list = [
                                                    [
                                                        label_list[0] - sentence_start,
                                                        label_list[1] - sentence_start,
                                                        label_list[2],
                                                    ]
                                                ]
                                            elif (
                                                label[0] > label_list[0]
                                                and label[1] <= label_list[1]
                                            ) or (
                                                label[0] >= label_list[0]
                                                and label[1] < label_list[1]
                                            ):
                                                sentence_label_lists[i].pop(j)
                                                sentence_label_list = [
                                                    [
                                                        label_list[0] - sentence_start,
                                                        label_list[1] - sentence_start,
                                                        label_list[2],
                                                    ]
                                                ]
                                            else:
                                                break
                                    else:
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
                        if (
                            from_id in label_data
                            and to_id in label_data
                            and relation_labels != []
                        ):
                            relational_label_list.append(
                                {
                                    "head": entity_info[from_id]["token_end"],
                                    "child": entity_info[to_id]["token_end"],
                                    "head_span": entity_info[from_id],
                                    "child_span": entity_info[to_id],
                                    "label": relation_labels[0],
                                }
                            )

            # Combine each sentence with its corresponding label_list
            for sent, sent_label_list in zip(sentences, sentence_label_lists):
                if sent_label_list != []:
                    training_data.append([sent, {"entities": sent_label_list}])

            # Combine text with its corresponding relation labels

            tokens = [
                {
                    "text": token.text,
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "ws": True if token.whitespace_ else False,
                    "id": i,
                }
                for i, token in enumerate(self.nlp(text))
            ]

            rel_data = {
                "text": text,
                "spans": spans,
                "meta": {"source": data["data"]["title"]}
                if not ground_truth
                else {"source": data["data"]["title"] + "_truth"},
                "tokens": tokens,
                "relations": relational_label_list,
                "answer": "accept",
            }

            relation_data.append(rel_data)

        return training_data, relation_data

    def preprocess_json_rel(
        self,
        relational_annotations_train: list,
        relational_annotations_val: list,
        save_path_train: str,
        save_path_dev,
    ):
        """
        Create training and validation datasets from a training set and store them as json files.

        :param training_data: The training data
        :param split_ratio: The ratio of the training data that is used for training
        :return: None
        """

        with open(save_path_train, "w") as file:
            # Save article text to file
            for annotation in relational_annotations_train:
                json.dump(annotation, file)
                file.write("\n")

        with open(save_path_dev, "w") as file:
            # Save article text to file
            for annotation in relational_annotations_val:
                json.dump(annotation, file)
                file.write("\n")

    def preprocess_json(
        self, training_data: list, validation_data: list, train_path: str, dev_path: str
    ):
        """
        Create training and validation datasets from a training set and store them as json files.

        :param training_data: The training data
        :param split_ratio: The ratio of the training data that is used for training
        :return: None
        """

        with open(train_path, "w") as file:
            # Save article text to file
            json.dump(training_data, file)

        with open(dev_path, "w") as file:
            # Save article text to file
            json.dump(validation_data, file)

    def preprocess_spacy(
        self, training_data: list, split_ratio: float = 0.8, warn: bool = False
    ):
        """
        Save the training and validation datasets as spacy files for model building.

        :param training_data: The training data
        :param split_ratio: The ratio of the training data that is used for training
        :param warn: A boolean that indicates if warnings should be shown
        :return: None
        """
        nlp = spacy.blank("en")

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

        train_save_path = os.path.join(ROOT_DIR, "ner_model/corpus", "train.spacy")
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

        dev_save_path = os.path.join(ROOT_DIR, "ner_model/corpus", "dev.spacy")
        dev_db.to_disk(dev_save_path)
