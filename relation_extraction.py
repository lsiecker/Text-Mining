import spacy
from thinc.types import Floats2d
from thinc.api import Model, Optimizer
from spacy.pipeline.trainable_pipe import TrainablePipe


def make_relation_extractor(
        nlp:Language, name:str, model:Model, *, threshold:float
):
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)

class RelationExtractor(TrainablePipe):
    def __init__(
            self,
            vocab:Vocab,
            model:Model,
            name:str="relationextractor",
            *,
            threshold:float
    ):
        self.model = model
        self.vocab = vocab
        self.name = name
        self._optimizer = None
        self.cfg = {"labels":[], "threshold": threshold}

    def __call__(self, doc:Doc) -> Doc:
        total_instances = len(self.model.attrs['get_instances'](doc))
        if total_instances == 0:
            return doc
        Y = self.model.predict([doc])
        self.set_annotations(doc, Y)
        return doc
    
    def set_annotations(self, docs: Iterable[Doc], scores:Floats2d) -> None:
        counter = 0
        get_instances = self.model.attrs['get_instances']
        for doc in docs:
            for (ent1, ent2) in get_instances(doc):
                starts = (ent1.start, ent2.start)
                if starts not in doc._.rel:
                    doc._.rel[starts] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[starts][label] = scores[counter, j]
                counter += 1

    def update(self, docs:Iterable[Doc], *, drop: float = 0.0, sgd: Optional[Optimizer] = None, losses: Optional[Dict[str,float]] = None) -> Dict[str, float]:
        if losses is None:
            losses = {}

        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        total_instances = 0
        for doc in docs:
            total_instances += len(self.model.attrs['get_instances'](doc))
        if total_instances == 0:
            return losses

        docs = [doc.predicted for doc in docs]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(docs, predictions)
        backprop(gradient)

        if sgd is not None:
            self.model.finish_update(sgd)

        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses