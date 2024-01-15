import spacy
from spacy.scorer import get_ner_prf, PRFScore, Scorer
from collections import defaultdict
from pprint import pprint

def custom_confm_scorer(examples, **kwargs):
    """Compute micro-PRF and per-entity PRF scores for a sequence of examples."""
    # Copied from https://github.com/explosion/spaCy/blob/master/spacy/scorer.py, get_ner_prf 
    # Added calculations for support
    score_per_type = defaultdict(PRFScore)
    support = dict()
    for eg in examples:
        if not eg.y.has_annotation("ENT_IOB"):
            continue
        golds = {(e.label_, e.start, e.end) for e in eg.y.ents}
        for g in golds:
            if g[0] not in support.keys():
                support[g[0]] = 1
            else:
                support[g[0]] += 1
        align_x2y = eg.alignment.x2y
        for pred_ent in eg.x.ents:
            if pred_ent.label_ not in score_per_type:
                score_per_type[pred_ent.label_] = PRFScore()
            indices = align_x2y[pred_ent.start : pred_ent.end]
            if len(indices):
                g_span = eg.y[indices[0] : indices[-1] + 1]
                # Check we aren't missing annotation on this span. If so,
                # our prediction is neither right nor wrong, we just
                # ignore it.
                if all(token.ent_iob != 0 for token in g_span):
                    key = (pred_ent.label_, indices[0], indices[-1] + 1)
                    if key in golds:
                        score_per_type[pred_ent.label_].tp += 1
                        golds.remove(key)
                    else:
                        score_per_type[pred_ent.label_].fp += 1
        for label, start, end in golds:
            score_per_type[label].fn += 1
    totals = PRFScore()
    for key, prf in score_per_type.items():
        totals += prf
        print(key, vars(prf))
    print(vars(totals))
    for label,val in support.items():
        print(label, "support: ", str(val))
    if len(totals) > 0:
        return {
            "ents_p": totals.precision,
            "ents_r": totals.recall,
            "ents_f": totals.fscore,
            "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items()},
        }
    else:
        return {
            "ents_p": None,
            "ents_r": None,
            "ents_f": None,
            "ents_per_type": None,
        }

@spacy.registry.scorers("custom_confm_scorer") 
def make_custom_confm_scorer(): 
    return custom_confm_scorer 