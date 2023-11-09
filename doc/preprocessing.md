# Project Initialization and Data Preprocessing

## Initialization

### Requirements

To initialize the project, make sure you have the required dependencies installed. You can use the provided `requirements.txt` file to install them. Run the following command:

```bash
pip install -r requirements.txt
```

This will ensure that your environment has all the required dependencies installed.

## Data Preprocessing

### Component 1: Wikipedia Data

#### Data Format
The Wikipedia extracts are stored in files within the folders AA and AB. Below is an example of how a file looks:

```text
{"id": "12", "revid": "43475265", "url": "https://en.wikipedia.org/wiki?curid=12", "title": "Anarchism", "text": "Anarchism is a political philosophy and movement that is skeptical of all justifications for authority and seeks to abolish the institutions it claims maintain unnecessary coercion and hierarchy, typically including nation states, and capitalism. Anarchism advocates for the replacement of the state with stateless societies and voluntary free associations. As a historically left-wing movement, 
... 
essence. Marxists state that this contradiction was responsible for their inability to act. In the anarchist vision, the conflict between liberty and equality was resolved through coexistence and intertwining."}
```

#### Preprocessing Steps
1. Data Collection:
    Place your wikipedia data (multiple of the above shown lines in one file) in the `data/component_1/AA` or `data/component_1/AB` folder. The data should be in the format shown above.
2. Run the component_1.ipynb file to preprocess the data. This includes filtering on the landmarks and creating files that can be used in the follow up notebook for training the model.

### Component 2: Technical Language Data
#### Data Format
The input for Component 2 is the file `initiating-events-summary-2021.xlsx`.

#### Preprocessing Steps
1. Data Collection:
    Place the `initiating-events-summary-2021.xlsx` file in the `data/component_2` folder.
2. Run the component_2.ipynb file to preprocess the data. 

### Next Steps
Using the files that are created by these preprocessing steps, it is now possible to add this to a labeling tool, e.g. LabelStudio. The output of the labeling tool can then be used to train the model.