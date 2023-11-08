# Text-Mining
This repository contains the code for extracting information from texts using NLP techniques. The code extends the Spacy NER component with custom training data and adds a REL component which is inspired by [Explosion AI's REL component](https://github.com/explosion/projects/tree/v3/tutorials/rel_component)


## Project Structure
- **Data** (`data/`): Contains all the raw wikipedia texts and annotation files.
- **NER component** (`ner_model/`): Contains the trained NER model including the preprocessed training and validation files.
- **REL component** (`rel_model/`): Contains the trained REL model including the preprocessed training and validation files.

## Documentation


## Assignment Description
Copied from the assignment description of the course 2AMM30 at the Eindhoven University of Technology. The downloaded official assignment description can be found [here](./doc/2AMM30%20Assignment%20description%20AY23-24.pdf).

### Data description
In this assignment we will undertake the general challenge of extracting information from 2 different
sources:
1. An unsupervised (albeit structured) large corpus: the entirety of Wikipedia.
2. Event registration and documentation for nuclear powerplants in the US

#### Wikipedia
A Wikipedia dump is a copy of all of the content from Wikipedia. This includes all of the articles,
images, and other media files. We provide a somewhat stripped/cleaned version of Wikipedia which saves you a considerable amount of computing power required to start your project. This stripped
version has had tables, headers and other “non-text” removed.

#### Technical language data (Nuclear powerplants)
Language data from industry often poses unique challenges due to its specialized nature and domain-
specific terminology. Unlike generic text found on the internet, industry-specific language is often rife
with technical jargon, abbreviations, and context-dependent meanings. This requires a deep
understanding of the specific field to accurately process and interpret the data.
To be familiarize with this challenge, a second dataset is provided which describes a collection of
events regarding unexpected reactor trips at commercial nuclear power plants in the U.S. It contains
a selection of metadata, a short-hand description as well as a longer abstract describing the
occurrences in detail.

### Assignment
The overall goal consists of the following: Perform information extraction on these dataset. In the
end, you should end with a collection of triplets (**[subject, relation, object]**) which could populate a
Knowledge Graph.
Perhaps for a particular use-case or subdomain in these datasets.
- Identify and label entities and relations of interests
- Select appropriate data sources that (likely) contain that information
- Build working models that can extract this information
- Evaluate the performance of extraction in parts and as a whole