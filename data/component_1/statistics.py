# In this folder read each json file and get the statistics of the data, average length of the text, number of sentences, max length of text and min length of text.
# The info is looking like this:
# {"id": 54, "data": {"title": "Aachen Cathedral", "text": "Aachen Cathedral () is a Roman Catholic church in Aachen, Germany and the seat of the Roman Catholic Diocese of Aachen.One of the oldest cathedrals in Europe, it was constructed by order of Emperor Charlemagne, who was buried there in 814. From 936 to 1531, the Palatine Chapel saw the coronation of thirty-one German kings and twelve queens. The church has been the mother church of the Diocese of Aachen since 1930. In 1978, Aachen Cathedral was one of the first 12 items to be listed on the UNESCO list of World Heritage sites, because of its exceptional artistry, architecture, and central importance in the history of the Holy Roman Empire."}}
# Only the text is used for the statistics.

import json
import os

# Path to the data folder, the folder this file is in
data_folder = os.path.dirname(os.path.abspath(__file__))

# Path to the folder where the json files are
json_folder = os.path.join(data_folder, "subset_texts")

# Get all json files in the folder
json_files = [
    pos_json for pos_json in os.listdir(json_folder) if pos_json.endswith(".json")
]

text_lengths_sent = []
text_lengths_word = []

# For each json file get the statistics for all texts combined:
for json_file in json_files:
    # Open the json file
    with open(os.path.join(json_folder, json_file)) as json_file:
        data = json.load(json_file)
        # Get the data
        data = data["data"]
        # Get the text
        text = data["text"]
        # Split the text into sentences
        sentences = text.split(".")
        words = text.split(" ")
        text_lengths_sent.append([sentences])
        text_lengths_word.append([words])

# Make a list of the amount of sentences per text
sentences_per_text = []
for text in text_lengths_sent:
    sentences_per_text.append(len(text[0]))

# Make a list of the amount of words per text
words_per_text = []
for text in text_lengths_word:
    words_per_text.append(len(text[0]))

# Get the average amount of sentences per text
average_sentences_per_text = sum(sentences_per_text) / len(sentences_per_text)

# Get the average amount of words per text
average_words_per_text = sum(words_per_text) / len(words_per_text)

# Get the max amount of sentences per text
max_sentences_per_text = max(sentences_per_text)
min_sentences_per_text = min(sentences_per_text)

# Get the max amount of words per text
max_words_per_text = max(words_per_text)
min_words_per_text = min(words_per_text)

# Print the statistics
print("Average amount of sentences per text: " + str(average_sentences_per_text))
print("Max amount of sentences per text: " + str(max_sentences_per_text))
print("Min amount of sentences per text: " + str(min_sentences_per_text))
print("Average amount of words per text: " + str(average_words_per_text))
print("Max amount of words per text: " + str(max_words_per_text))
print("Min amount of words per text: " + str(min_words_per_text))


# Do the same but then not for all json files, but for one txt file with on each line a text. This file can be found in ../data/component_2/initiating-events-summary-2021.txt

# Path to the folder where the txt file is
txt_folder = os.path.join("data\component_2\initiating-events-summary-2021.txt")

text_lengths_sent_2 = []
text_lengths_word_2 = []

# Get all txt files in the file, every line is a text
with open(txt_folder) as f:
    texts = f.readlines()
    for text in texts:
        # Split the text into sentences
        sentences = text.split(".")
        words = text.split(" ")
        text_lengths_sent_2.append([sentences])
        text_lengths_word_2.append([words])


# Make a list of the amount of sentences per text
sentences_per_text_2 = []
for text in text_lengths_sent_2:
    sentences_per_text_2.append(len(text[0]))

# Make a list of the amount of words per text
words_per_text_2 = []
for text in text_lengths_word_2:
    words_per_text_2.append(len(text[0]))

# Get the average amount of sentences per text
average_sentences_per_text_2 = sum(sentences_per_text_2) / len(sentences_per_text_2)

# Get the average amount of words per text
average_words_per_text_2 = sum(words_per_text_2) / len(words_per_text_2)

# Get the max amount of sentences per text
max_sentences_per_text_2 = max(sentences_per_text_2)
min_sentences_per_text_2 = min(sentences_per_text_2)

# Get the max amount of words per text
max_words_per_text_2 = max(words_per_text_2)
min_words_per_text_2 = min(words_per_text_2)

# Print the statistics
print("Average amount of sentences per text: " + str(average_sentences_per_text_2))
print("Max amount of sentences per text: " + str(max_sentences_per_text_2))
print("Min amount of sentences per text: " + str(min_sentences_per_text_2))
print("Average amount of words per text: " + str(average_words_per_text_2))
print("Max amount of words per text: " + str(max_words_per_text_2))
print("Min amount of words per text: " + str(min_words_per_text_2))


# Create a grid of 3x2 subplots and plot the statistics in the subplots. There is a component 1 and 2, 2 is indicated with a _2 in the variable name
# create one bar chart for e.g. text_lengths_sent and text_lenghts_sent_2 to compare them and do this for the avg, max and min values of sentences and words

import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 3, figsize=(10, 6))

# Plotting sentences data
axs[0, 0].bar(
    ["Component 1", "Component 2"],
    [average_sentences_per_text, average_sentences_per_text_2],
)
axs[0, 0].set_title("Average sentences per text")
axs[0, 2].bar(
    ["Component 1", "Component 2"],
    [max_sentences_per_text, max_sentences_per_text_2],
)
axs[0, 2].set_title("Max sentences per text")
axs[0, 1].bar(
    ["Component 1", "Component 2"],
    [min_sentences_per_text, min_sentences_per_text_2],
)
axs[0, 1].set_title("Min sentences per text")

# Plotting words data
axs[1, 0].bar(
    ["Component 1", "Component 2"],
    [average_words_per_text, average_words_per_text_2],
)
axs[1, 0].set_title("Average words per text")
axs[1, 2].bar(
    ["Component 1", "Component 2"],
    [max_words_per_text, max_words_per_text_2],
)
axs[1, 2].set_title("Max words per text")
axs[1, 1].bar(
    ["Component 1", "Component 2"],
    [min_words_per_text, min_words_per_text_2],
)
axs[1, 1].set_title("Min words per text")

# Adjust layout
plt.tight_layout(pad=3.0)

# Add labels and rounded numbers on top of bars
for ax in axs.flat:
    ax.set(xlabel="", ylabel="Amount")

    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(
            f"{height:.2f}", (x + width / 2, y + height / 2), ha="center", va="center"
        )

# make the second bar in the chart for component 2 red
for i in range(2):
    for j in range(3):
        axs[i, j].patches[1].set_facecolor("green")


# Show the plot
plt.show()
