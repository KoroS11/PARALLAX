# ruff: noqa
import json, sys, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load(fn):
    return [json.loads(l) for l in open(fn, encoding="utf-8") if l.strip()]


S = {n: load(f"data/splits/{n}.jsonl") for n in ("dev", "val", "held_out_v2")}


def XY(rows, vec=None, fit=False):
    t = [r["text"] for r in rows]
    y = np.array([int(r["label_binary"]) for r in rows])
    if vec is None:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    X = vec.fit_transform(t) if fit else vec.transform(t)
    return X, y, vec


print("== 1. same-split CV AUC (random vs grouped by family/technique) ==")
for n, rows in S.items():
    X, y, _ = XY(rows, fit=True)
    p = cross_val_predict(
        LogisticRegression(max_iter=1000),
        X,
        y,
        cv=StratifiedKFold(5, shuffle=True, random_state=0),
        method="predict_proba",
    )[:, 1]
    line = f"{n:12} random-CV AUC {roc_auc_score(y, p):.3f}"
    gkey = (
        "family"
        if "family" in rows[0]
        else ("technique" if "technique" in rows[0] else None)
    )
    if gkey:
        groups = [r.get(gkey, "unknown") for r in rows]
        if len(set(groups)) >= 5:
            try:
                pg = cross_val_predict(
                    LogisticRegression(max_iter=1000),
                    X,
                    y,
                    cv=GroupKFold(5),
                    groups=groups,
                    method="predict_proba",
                )[:, 1]
                line += f" | grouped-by-{gkey} AUC {roc_auc_score(y, pg):.3f}"
            except Exception as e:
                line += f" | grouped failed: {e}"
    print(line)

print(
    "\n== 2. cross-split transfer: train dev+val, test held_out_v2 (AUC; 0.5 = no shortcut) =="
)
tr = S["dev"] + S["val"]
Xtr, ytr, vec = XY(tr, fit=True)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
Xte, yte, _ = XY(S["held_out_v2"], vec=vec)
print("transfer AUC:", round(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]), 3))

print("\n== 3. nearest-neighbour twins within each split ==")
for n, rows in S.items():
    X, y, _ = XY(rows, fit=True)
    sim = cosine_similarity(X)
    np.fill_diagonal(sim, -1)
    nn = sim.argmax(1)
    opp = float(np.mean(y[nn] != y))
    msim = float(np.mean(sim.max(1)))
    print(
        f"{n:12} NN has opposite label: {opp:.2f}   mean NN similarity: {msim:.2f}   share with sim>0.8: {np.mean(sim.max(1) > 0.8):.2f}"
    )

print("\n== sample twins (held_out_v2) ==")
rows = S["held_out_v2"]
X, y, _ = XY(rows, fit=True)
sim = cosine_similarity(X)
np.fill_diagonal(sim, -1)
idx = np.argsort(-sim.max(1))[:3]
for i in idx:
    j = sim[i].argmax()
    print(
        f"label {y[i]} vs {y[j]} (sim {sim[i, j]:.2f})\n  A: {rows[i]['text'][:150]!r}\n  B: {rows[j]['text'][:150]!r}"
    )
