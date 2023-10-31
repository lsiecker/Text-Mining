"""Convert entity annotation from spaCy v2 TRAIN_DATA format to spaCy v3
.spacy format."""
import srsly
from tqdm import tqdm
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin
from rel_pipe import make_relation_extractor, LABELS


def convert(lang: str, input_path: Path, output_path: Path):
    nlp = spacy.blank(lang)  # empty nlp object initialized
    db = DocBin()  # store processed text data, often in binary format
    for text, annot in tqdm(srsly.read_json(input_path)):
        doc = nlp.make_doc(text)
        doc.set_extension("rel", default={}, force=True)
        ents = []
        relations = {}

        relation_entity_list = []

        for entity in annot["entities"]:
            for start, end, label in entity:
                span = doc.char_span(start, end, label=label, alignment_mode="contract")

                if span is None:
                    msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                    # warnings.warn(msg)
                else:
                    ents.append(span)

        for rel in annot["relations"]:
            head_span, child_span, label = (
                rel["head_span"],
                rel["child_span"],
                rel["label"],
            )
            # Add the entity to the list of entities
            if head_span not in relation_entity_list:
                relation_entity_list.append(head_span)
            if child_span not in relation_entity_list:
                relation_entity_list.append(child_span)

        # add all combinations of entities to the relations dictionary
        for i in range(len(relation_entity_list)):
            for j in range(len(relation_entity_list)):
                if i != j:
                    relations[(i, j)] = {}
                    for label in LABELS:
                        relations[(i, j)][label] = 0.0

        for rel in annot["relations"]:
            head_span, child_span, label = (
                rel["head_span"],
                rel["child_span"],
                rel["label"],
            )
            if (
                head_span["token_start"] is None
                or child_span["token_start"] is None
                or head_span["token_start"] == child_span["token_start"]
            ):
                msg = f"Skipping relation [{head_span}, {child_span}, {label}] in the following text because the character span '{doc.text[head_span['start']: head_span['end']]}' or '{doc.text[child_span['start']: child_span['end']]}' does not align with token boundaries:\n\n{repr(text)}\n"
                # warnings.warn(msg)
            else:
                relations[(head_span["token_start"], child_span["token_start"])][
                    label
                ] = 1.0

        try:
            doc.ents = ents
        except ValueError as e:
            msg = f"Skipping the following text because of a problem with the entities: {repr(text)}\n\n{e}"
            warnings.warn(msg)
        try:
            doc._.rel = relations
        except ValueError as e:
            msg = f"Skipping the following text because of a problem with the relations: {repr(text)}\n\n{e}"
            warnings.warn(msg)
        db.add(doc)
    db.to_disk(output_path)


if __name__ == "__main__":
    typer.run(convert)
