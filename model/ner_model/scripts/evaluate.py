import spacy
from spacy.scorer import get_ner_prf, PRFScore, Scorer
from collections import defaultdict
from pprint import pprint

def custom_confm_scorer(examples, **kwargs):
    """Compute micro-PRF and per-entity PRF scores for a sequence of examples."""
    score_per_type = defaultdict(PRFScore)
    for eg in examples:
        total_pred = 0
        sent = len(eg.x)
        if not eg.y.has_annotation("ENT_IOB"):
            continue
        golds = {(e.label_, e.start, e.end, e) for e in eg.y.ents}
        align_x2y = eg.alignment.x2y
        for pred_ent in eg.x.ents:
            #print(pred_ent, pred_ent.label_)
            if pred_ent.label_ not in score_per_type:
                score_per_type[pred_ent.label_] = PRFScore()
            indices = align_x2y[pred_ent.start : pred_ent.end]
            if len(indices):
                g_span = eg.y[indices[0] : indices[-1] + 1]
                # Check we aren't missing annotation on this span. If so,
                # our prediction is neither right nor wrong, we just
                # ignore it.
                if all(token.ent_iob != 0 for token in g_span):
                    key = (pred_ent.label_, indices[0], indices[-1] + 1, pred_ent)
                    print(dir(pred_ent))
                    total_pred+=len(pred_ent.ents)
                    if key in golds:
                        score_per_type[pred_ent.label_].tp += 1
                        golds.remove(key)
                    else:
                        score_per_type[pred_ent.label_].fp += 1
        for label, start, end, e in golds:
            total_pred+=len(e.ents)
            score_per_type[label].fn += 1
    totals = PRFScore()
    for key, prf in score_per_type.items():
        totals += prf
       # print(key, vars(prf))
    scorer = Scorer()
    res = scorer.score(examples)
    #print(vars(totals))
    #print(len(examples))
    print(total_pred) #number of predictions
    print(sent) #sentence length
    tn = sent - total_pred
    return res

@spacy.registry.scorers("custom_confm_scorer") 
def make_custom_confm_scorer(): 
    return custom_confm_scorer 