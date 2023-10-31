# make the factory work
from scripts.rel_pipe import make_relation_extractor

# make the config work
from scripts.rel_model import (
    create_relation_model,
    create_classification_layer,
    create_instances,
    create_tensors,
)
