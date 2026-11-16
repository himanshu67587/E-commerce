"""Part 2 - classification and evaluation (train 560 / test 240)."""
import numpy as np
from common import (load_data, split_and_standardise, fit_logistic, predict_proba, bce,
                    confusion, metrics, knn_cv, knn_predict, write_results, write_csv, FEATURES)

X, y, value = load_data()
tr, te, Ztr, Zte, mu, s = split_and_standardise(X, y)
ytr, yte = y[tr], y[te]
R = dict(train_rows=len(tr), train_positives=int(ytr.sum()),
         test_rows=len(te), test_positives=int(yte.sum()))

# 2.2 logistic regression from scratch
w, b, L = fit_logistic(Ztr, ytr, 0.1, 1000, record=(0, 1, 10, 100, 1000))
for e in (0, 1, 10, 100, 1000):
    R[f"loss_epoch_{e}"] = L[e]
for j, name in enumerate(FEATURES):
    R[f"weight_{name}"] = w[j]
R["bias"] = b
ptest = predict_proba(Zte, w, b)
ptrain = predict_proba(Ztr, w, b)
R["test_bce"] = bce(yte, ptest)

# 2.3 learning-rate effect
for eta in (0.01, 0.1, 1.0):
    _, _, l = fit_logistic(Ztr, ytr, eta, 200, record=(200,))
    R[f"loss_epoch_200_eta_{eta}"] = l[200]

# 2.4 odds ratios
for j, name in enumerate(FEATURES):
    R[f"odds_ratio_{name}"] = float(np.exp(w[j]))

# 2.5 two thresholds
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, (ptest >= tau).astype(int))
    acc, prec, rec, f1, spec = metrics(tp, fp, tn, fn)
    for k, v in zip(["tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"],
                    [tp, fp, tn, fn, acc, prec, rec, f1, spec]):
        R[f"tau_{tau}_{k}"] = v
R["blind_accuracy"] = float(np.mean(yte == 0))

# 2.6 threshold sweep
rows = []
for k in range(1, 10):
    tau = k / 10
    tp, fp, tn, fn = confusion(yte, (ptest >= tau).astype(int))
    rows.append([round(tau, 1), tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])
write_csv("outputs/part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# 2.7 KNN with 5-fold CV
cv, best_k = knn_cv(Ztr, ytr)
write_csv("outputs/part2_knn_cv.csv", ["K", "mean_cv_f1"], cv)
R["knn_best_k"] = best_k
pred, _ = knn_predict(Ztr, ytr, Zte, best_k)
tp, fp, tn, fn = confusion(yte, pred)
acc, prec, rec, f1, _ = metrics(tp, fp, tn, fn)
for k, v in zip(["tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1"],
                [tp, fp, tn, fn, acc, prec, rec, f1]):
    R[f"knn_{k}"] = v

# 2.8 threshold chosen with money (training rows only)
def cost_at(yv, p, tau):
    _, fp, _, fn = confusion(yv, (p >= tau).astype(int))
    return 50 * fp + 400 * fn

taus = [k / 20 for k in range(1, 20)]
train_costs = [cost_at(ytr, ptrain, t) for t in taus]
best_i = int(np.argmin(train_costs))  # first minimum -> smaller tau on ties
tau_star = taus[best_i]
R["cost_tau"] = tau_star
R["cost_train_at_tau"] = train_costs[best_i]
R["cost_test_at_tau"] = cost_at(yte, ptest, tau_star)
R["cost_test_at_0.5"] = cost_at(yte, ptest, 0.5)
R["cost_test_flag_nobody"] = 400 * int(yte.sum())

write_results("outputs/part2_results.txt", R)
