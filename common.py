"""Shared helpers (data loading, split, logistic regression, KNN, metrics, file writers).
Importing this file does not run any Part."""
import csv
import math
import numpy as np

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]


def load_data(path="data/records.csv"):
    """Return X (800x7), outcome (800,), value (800,) in file order."""
    with open(path) as f:
        rows = list(csv.reader(f))[1:]
    arr = np.array([[float(v) for v in r[1:]] for r in rows])
    return arr[:, :7], arr[:, 7].astype(int), arr[:, 8]


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4f}"
    return str(v)


def write_results(path, d):
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k} = {fmt(v)}\n")


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def sd(a):
    """Sample standard deviation (divide by n - 1)."""
    return math.sqrt(np.sum((a - a.mean()) ** 2) / (len(a) - 1))


# ---------- Task 2.1: split and standardise ----------
def split_and_standardise(X, y):
    train_idx, test_idx = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = (7 * len(idx)) // 10
        train_idx += list(idx[:k])
        test_idx += list(idx[k:])
    train_idx, test_idx = np.array(sorted(train_idx)), np.array(sorted(test_idx))
    mu = X[train_idx].mean(axis=0)
    s = np.array([sd(X[train_idx][:, j]) for j in range(X.shape[1])])
    Ztr, Zte = (X[train_idx] - mu) / s, (X[test_idx] - mu) / s
    return train_idx, test_idx, Ztr, Zte, mu, s


# ---------- Task 2.2: logistic regression ----------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_logistic(Z, y, eta=0.1, epochs=1000, record=()):
    """Batch gradient descent from zero. Returns w, b and {epoch: loss} for epochs in record."""
    n, d = Z.shape
    w, b = np.zeros(d), 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(Z @ w + b))
    for e in range(1, epochs + 1):
        p = sigmoid(Z @ w + b)
        err = p - y
        w = w - eta * (Z.T @ err) / n
        b = b - eta * err.sum() / n
        if e in record:
            losses[e] = bce(y, sigmoid(Z @ w + b))
    return w, b, losses


def predict_proba(Z, w, b):
    return sigmoid(Z @ w + b)


# ---------- metrics ----------
def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    spec = safe_div(tn, tn + fp)
    return acc, prec, rec, f1, spec


# ---------- Task 2.7: KNN ----------
def knn_share(ref_Z, ref_y, query_Z, K):
    """Share of positive neighbours among the K nearest (ties broken by file order)."""
    d2 = ((query_Z[:, None, :] - ref_Z[None, :, :]) ** 2).sum(axis=2)
    order = np.argsort(d2, axis=1, kind="stable")[:, :K]
    return ref_y[order].mean(axis=1)


def knn_predict(ref_Z, ref_y, query_Z, K):
    share = knn_share(ref_Z, ref_y, query_Z, K)
    pos_count = np.rint(share * K).astype(int)
    return (pos_count >= (K + 1) // 2).astype(int), share


def knn_cv(Ztr, ytr, ks=range(1, 22, 2), folds=5):
    fold = np.arange(len(ytr)) % folds
    result = []
    for K in ks:
        f1s = []
        for f in range(folds):
            ref, hold = fold != f, fold == f
            pred, _ = knn_predict(Ztr[ref], ytr[ref], Ztr[hold], K)
            f1s.append(metrics(*confusion(ytr[hold], pred))[3])
        result.append((K, float(np.mean(f1s))))
    best = max(result, key=lambda t: (t[1], -t[0]))[0]  # ties -> smaller K
    return result, best
