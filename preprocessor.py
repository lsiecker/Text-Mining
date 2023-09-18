import ftfy
import spacy
import os
import json
from tqdm import tqdm

import re

import nltk
from nltk import word_tokenize, pos_tag, ne_chunk

class Preprocessor:
    def __init__(self, basedir, spacy_lib = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_lib)
        self.basedir = basedir

    def fix_unicode(self, data):
        print("Cleaning data with ftfy...")
        output = []
        for article in tqdm(data):
            text = article['text']
            unicode_fix = ftfy.fix_text(text)
            article['fixed_text'] = unicode_fix
            output.append(article)
        return output 
    
    def load_file(self, name):
        file_path = os.path.join(self.basedir, 'data\\', name)
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    
    def save_file(self, data, folder):
        for i, article in enumerate(data):
            title = ''.join(char for char in article['title'].lower() if char.isalnum()) + ".json"
            file_path = os.path.join(self.basedir, 'data\\', folder, title)
            text = re.sub(r'[^\x00-\x7F]+', '', article['text'])
            to_dump = {
                "id" : i,
                "data" : {
                    "title" : article['title'],
                    "text" : text,
                }
            }
            with open(file_path, "w") as file:
                # Save article text to file
                json.dump(to_dump, file)
    
    def clean_alphanumeric(self, data, pattern = r'\W+'):
        print("Cleaning data with given regex...")
        cleaned_articles = []

        # Loop through each text in the list
        for article in tqdm(data):
            text = article['text']
            # Tokenize the text using the regular expression pattern
            words = re.split(pattern, text)
            # Count occurrences of non-alphanumeric words
            char_words = [word for word in words if word.isalnum()]
            article['cleaned_text'] = " ".join(char_words)
            cleaned_articles.append(article)
        return cleaned_articles
    
    def ner_spacy(self, text):
        doc = self.nlp(text)
        return doc
    
    def ner_nltk(self, text):
        return ne_chunk(pos_tag(word_tokenize(text)))