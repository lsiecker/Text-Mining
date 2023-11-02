import json

import typer
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer

msg = Printer()

SYMM_LABELS = ["Binds"]
MAP_LABELS = {
    "org:created_by": "created_by",
    "org:located_in": "located_in",
    "org:happened_on": "happened_on",
    "org:has_occupation": "has_occupation",
    "org:is_condition": "is_condition",
    "org:enlisted_in": "enlisted_in",
    "org:is_type": "is_type",
    "org:has_component": "has_component",
    "org:is_similar_to": "is_similar_to",
    "org:unrelated": "unrelated",
}


def main(json_loc: Path, train_file: Path, dev_file: Path):
    """Creating the corpus from the Prodigy annotations."""
    Doc.set_extension("rel", default={})
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": []}
    ids = {"train": set(), "dev": set(), "test": set()}
    count_all = {"train": 0, "dev": 0, "test": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0}

    with json_loc.open("r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            if example["answer"] == "accept":
                neg = 0
                pos = 0
                try:
                    # Parse the tokens
                    words = [t["text"] for t in example["tokens"]]
                    spaces = [t["ws"] for t in example["tokens"]]
                    doc = Doc(vocab, words=words, spaces=spaces)

                    # Parse the GGP entities
                    spans = example["spans"]
                    entities = []
                    span_end_to_start = {}
                    for span in spans:
                        entity = doc.char_span(
                            span["start"], span["end"], label=span["label"]
                        )
                        span_end_to_start[span["token_end"]] = span["token_start"]
                        if entity is not None and entity not in entities:
                            entities.append(entity)
                        span_starts.add(span["token_start"])
                        try:
                            doc.ents = entities
                        except ValueError as e:
                            entities.pop()
                            print(e)

                    # Parse the relations
                    rels = {}
                    for x1 in span_starts:
                        for x2 in span_starts:
                            rels[(x1, x2)] = {}
                    relations = example["relations"]
                    for relation in relations:
                        # the 'head' and 'child' annotations refer to the end token in the span
                        # but we want the first token
                        start = span_end_to_start[relation["head"]]
                        end = span_end_to_start[relation["child"]]
                        label = relation["label"]
                        label = MAP_LABELS[label]
                        if label not in rels[(start, end)]:
                            rels[(start, end)][label] = 1.0
                            pos += 1
                        if label in SYMM_LABELS:
                            if label not in rels[(end, start)]:
                                rels[(end, start)][label] = 1.0
                                pos += 1

                    # The annotation is complete, so fill in zero's where the data is missing
                    for x1 in span_starts:
                        for x2 in span_starts:
                            for label in MAP_LABELS.values():
                                if label not in rels[(x1, x2)]:
                                    neg += 1
                                    rels[(x1, x2)][label] = 0.0
                    doc._.rel = rels

                    # only keeping documents with at least 1 positive case
                    if pos > 0:
                        # use the original PMID/PMCID to decide on train/dev/test split
                        article_id = example["meta"]["source"]
                        article_id = article_id.replace(
                            "BioNLP 2011 Genia Shared Task, ", ""
                        )
                        article_id = article_id.replace(".txt", "")
                        article_id = article_id.split("_")[-1]
                        if article_id.endswith("_truth"):
                            ids["dev"].add(article_id)
                            docs["dev"].append(doc)
                            count_pos["dev"] += pos
                            count_all["dev"] += pos + neg
                        else:
                            ids["train"].add(article_id)
                            docs["train"].append(doc)
                            count_pos["train"] += pos
                            count_all["train"] += pos + neg
                except KeyError as e:
                    msg.fail(
                        f"Skipping doc because of key error: {e} in {example['meta']['source']}"
                    )

    docbin = DocBin(docs=docs["train"], store_user_data=True)
    docbin.to_disk(train_file)
    msg.info(
        f"{len(docs['train'])} training sentences from {len(ids['train'])} articles, "
        f"{count_pos['train']}/{count_all['train']} pos instances."
    )

    docbin = DocBin(docs=docs["dev"], store_user_data=True)
    docbin.to_disk(dev_file)
    msg.info(
        f"{len(docs['dev'])} dev sentences from {len(ids['dev'])} articles, "
        f"{count_pos['dev']}/{count_all['dev']} pos instances."
    )


if __name__ == "__main__":
    typer.run(main)
