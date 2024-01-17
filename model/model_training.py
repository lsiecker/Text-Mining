import os
import subprocess
from pathlib import Path

import spacy
from spacy.cli.train import train
from spacy.cli.evaluate import evaluate
from ner_model.scripts.convert import convert as ner_convert

from processor import Processor

# Set the root directory of the project
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname("model_trainig.ipynb"), os.pardir)
)  # This file is the root of the project
DATA_PATH = os.path.join(ROOT_DIR, "data")

# Determine which assignment component to run
#############################################
COMPONENT: int = 2  # <<<=== 1 or 2
#############################################

# Define all general paths
NER_MODEL: Path = os.path.join(ROOT_DIR, "model", "ner_model")
REL_MODEL: Path = os.path.join(ROOT_DIR, "model", "rel_model")

processor = Processor(ROOT_DIR)

# Define all component specific paths
if COMPONENT == 1:
    ANNOTATIONS: Path = "final_assignment_1.json"
    GROUND_TRUTHS: Path = "label_studio_ground_truth_task1.json"
    NER_DEV: Path = os.path.join(NER_MODEL, "assets", "dev_1.json")
    NER_TRAIN: Path = os.path.join(NER_MODEL, "assets", "train_1.json")
    NER_DEV_CORP: Path = os.path.join(NER_MODEL, "corpus", "dev_1.spacy")
    NER_TRAIN_CORP: Path = os.path.join(NER_MODEL, "corpus", "train_1.spacy")
    NER_OUTPUT: Path = os.path.join(NER_MODEL, "training_1")
    REL_ASSETS_TRAIN: Path = os.path.join(
        REL_MODEL, "assets", "annotations_1_train.jsonl"
    )
    REL_ASSETS_DEV: Path = os.path.join(REL_MODEL, "assets", "annotations_1_dev.jsonl")
    REL_OUTPUT: Path = os.path.join(REL_MODEL, "training_1")
elif COMPONENT == 2:
    ANNOTATIONS: Path = "final_assignment_2.json"
    GROUND_TRUTHS: Path = "label_studio_ground_truth_task2.json"
    NER_DEV: Path = os.path.join(NER_MODEL, "assets", "dev_2.json")
    NER_TRAIN: Path = os.path.join(NER_MODEL, "assets", "train_2.json")
    NER_DEV_CORP: Path = os.path.join(NER_MODEL, "corpus", "dev_2.spacy")
    NER_TRAIN_CORP: Path = os.path.join(NER_MODEL, "corpus", "train_2.spacy")
    NER_OUTPUT: Path = os.path.join(NER_MODEL, "training_2")
    REL_ASSETS_TRAIN: Path = os.path.join(
        REL_MODEL, "assets", "annotations_2_train.jsonl"
    )
    REL_ASSETS_DEV: Path = os.path.join(REL_MODEL, "assets", "annotations_2_dev.jsonl")
    REL_OUTPUT: Path = os.path.join(REL_MODEL, "training_2")
else:
    raise ValueError("COMPONENT must be 1 or 2")

export_data = processor.loadFile(ANNOTATIONS)

# Filter out annotations for which a ground truth exists (drop other annotations for this article as well)
training_data_export = [
    item
    for item in export_data
    if all(annotation["ground_truth"] is False for annotation in item["annotations"])
]
ground_truth_export = processor.loadFile(GROUND_TRUTHS)

training_data, training_relations = processor.process_export_sentences(
    training_data_export, component=COMPONENT
)
validation_data, validation_relations = processor.process_export_sentences(
    ground_truth_export, ground_truth=True, component=COMPONENT
)

processor.preprocess_json(
    training_data=training_data,
    validation_data=validation_data,
    train_path=NER_TRAIN,
    dev_path=NER_DEV,
)
processor.preprocess_json_rel(
    relational_annotations_train=training_relations,
    relational_annotations_val=validation_relations,
    save_path_train=REL_ASSETS_TRAIN,
    save_path_dev=REL_ASSETS_DEV,
)

################# NER MODEL #################
ner_convert("en", NER_TRAIN, NER_TRAIN_CORP)
ner_convert("en", NER_DEV, NER_DEV_CORP)


train(
    "ner_model/configs/config.cfg",
    output_path=NER_OUTPUT,
    overrides={"paths.train": NER_TRAIN_CORP, "paths.dev": NER_DEV_CORP},
)

evaluate(
    os.path.join(NER_OUTPUT, "model-best"),
    NER_DEV_CORP,
    output=os.path.join(NER_OUTPUT, "metrics.json"),
)

################# REL MODEL #################
if spacy.prefer_gpu():
    output = subprocess.run(
        f"spacy project run all_{COMPONENT}_gpu", cwd="rel_model", capture_output=True
    )
else:
    output = subprocess.run(
        f"spacy project run all_{COMPONENT}", cwd="rel_model", capture_output=True
    )
