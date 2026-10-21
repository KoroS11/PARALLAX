# ruff: noqa
import glob, json, re, random, collections, statistics, sys

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

splits = {}
for fn in sorted(glob.glob("data/splits/*.jsonl")):
    rows = [json.loads(l) for l in open(fn, encoding="utf-8") if l.strip()]
    splits[fn] = rows

pats = {
    "CANARY": r"CANARY",
    "mock_tool()": r"mock_\w+\(",
    "[Action:": r"\[Action:",
    "ignore/override": r"ignore|override",
}
random.seed(1)
for fn, rows in splits.items():
    y = [int(r.get("label_binary", 0)) for r in rows]
    print(
        "\n==", fn, "n =", len(rows), "attacks =", sum(y), "benign =", len(y) - sum(y)
    )
    for name, p in pats.items():
        a = sum(bool(re.search(p, r["text"], re.I)) for r, l in zip(rows, y) if l == 1)
        b = sum(bool(re.search(p, r["text"], re.I)) for r, l in zip(rows, y) if l == 0)
        print(f"  {name:16} attack {a}/{sum(y)}  benign {b}/{len(y) - sum(y)}")
    la = [len(r["text"]) for r, l in zip(rows, y) if l == 1]
    lb = [len(r["text"]) for r, l in zip(rows, y) if l == 0]
    if la and lb:
        print(
            "  median length attack",
            statistics.median(la),
            "benign",
            statistics.median(lb),
        )
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score

        if 0 < sum(y) < len(y) and min(sum(y), len(y) - sum(y)) >= 5:
            X = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), min_df=2
            ).fit_transform([r["text"] for r in rows])
            p = cross_val_predict(
                LogisticRegression(max_iter=1000), X, y, cv=5, method="predict_proba"
            )[:, 1]
            print("  char-ngram LR CV AUC:", round(roc_auc_score(y, p), 3))
    except Exception as e:
        print("  AUC skipped:", e)
    for lab, name in ((1, "ATTACK"), (0, "BENIGN")):
        pool = [r["text"][:160].replace("\n", " ") for r, l in zip(rows, y) if l == lab]
        for t in random.sample(pool, min(3, len(pool))):
            print(f"  [{name}] {t}")

for fn in glob.glob("eval/logs/latest_judge*.jsonl"):
    c = collections.Counter(
        json.loads(l)["split"] for l in open(fn, encoding="utf-8") if l.strip()
    )
    print("\njudge log", fn, dict(c))
