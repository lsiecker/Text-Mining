import srsly
import typer
import warnings
from pathlib import Path

import spacy
from spacy.tokens import DocBin


def convert(lang: str, input_path: Path, output_path: Path):
    """
    Convert the json document in the assets folder (input_path) to a .spacy file that can be used for training.

    Parameters
    ----------
    lang: str
        Language of the data
    input_path: Path
        Path to the input file
    output_path: Path
        Path to the output file
    """
    nlp = spacy.blank(lang)
    db = DocBin()
    print(f"Start converting NER data...")
    for text, annot in srsly.read_json(input_path):
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)
    print(f"Finished convertin NER data")


if __name__ == "__main__":
    typer.run(convert)
