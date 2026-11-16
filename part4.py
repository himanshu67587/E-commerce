"""Part 4 - model choice, boosting arithmetic and architecture sizing."""
import math
import numpy as np
from common import (load_data, split_and_standardise, fit_logistic, predict_proba, confusion,
                    metrics, knn_cv, knn_predict, write_results, write_csv)

X, y, value = load_data()
tr, te, Ztr, Zte, mu, s = split_and_standardise(X, y)
ytr, yte = y[tr], y[te]
Xtr, Xte = X[tr], X[te]          # original units
R = {}


# 4.1 which split is purest?
def entropy(P, n):
    if n == 0:
        return 0.0
    e = 0.0
    for p in (P / n, 1 - P / n):
        if p > 0:
            e -= p * math.log2(p)
    return e


def gini(P, n):
    return 0.0 if n == 0 else 1 - (P / n) ** 2 - (1 - P / n) ** 2


N, P = len(ytr), int(ytr.sum())
R["root_entropy"], R["root_gini"] = entropy(P, N), gini(P, N)
splits = {"S1": Xtr[:, 5] == 1, "S2": Xtr[:, 6] == 1, "S3": Xtr[:, 2] > 0.5, "S4": Xtr[:, 0] <= 24}
rows, ig, gg = [], {}, {}
for name, yes in splits.items():
    ny, py = int(yes.sum()), int(ytr[yes].sum())
    nn, pn = N - ny, P - py
    ig[name] = entropy(P, N) - (ny / N * entropy(py, ny) + nn / N * entropy(pn, nn))
    gg[name] = gini(P, N) - (ny / N * gini(py, ny) + nn / N * gini(pn, nn))
    rows.append([name, ny, py, nn, pn, ig[name], gg[name]])
write_csv("outputs/part4_splits.csv",
          ["split", "n_yes", "positives_yes", "n_no", "positives_no", "info_gain", "gini_gain"], rows)
best_ig = max(ig, key=lambda k: (ig[k], -int(k[1])))
best_gg = max(gg, key=lambda k: (gg[k], -int(k[1])))
R["best_split_by_info_gain"], R["best_split_by_gini_gain"] = best_ig, best_gg

# 4.2 stump + ensembles on the test rows
yes_tr = splits[best_ig]
R["stump_rate_yes"] = float(ytr[yes_tr].mean())
R["stump_rate_no"] = float(ytr[~yes_tr].mean())
cond = {"S1": lambda A: A[:, 5] == 1, "S2": lambda A: A[:, 6] == 1,
        "S3": lambda A: A[:, 2] > 0.5, "S4": lambda A: A[:, 0] <= 24}[best_ig]
p_stump = np.where(cond(Xte), R["stump_rate_yes"], R["stump_rate_no"])

w, b, _ = fit_logistic(Ztr, ytr, 0.1, 1000)
p_log = predict_proba(Zte, w, b)
_, best_k = knn_cv(Ztr, ytr)
knn_pred, p_knn = knn_predict(Ztr, ytr, Zte, best_k)

pred_log = (p_log >= 0.3).astype(int)
pred_stump = (p_stump >= 0.3).astype(int)
pred_hard = ((pred_log + knn_pred + pred_stump) >= 2).astype(int)
pred_soft = (((p_log + p_knn + p_stump) / 3) >= 0.3).astype(int)
vrows = []
for name, pr in [("logistic", pred_log), ("knn", knn_pred), ("stump", pred_stump),
                 ("hard_vote", pred_hard), ("soft_vote", pred_soft)]:
    tp, fp, tn, fn = confusion(yte, pr)
    _, prec, rec, f1, _ = metrics(tp, fp, tn, fn)
    vrows.append([name, tp, fp, tn, fn, prec, rec, f1])
write_csv("outputs/part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

# 4.3 two rounds of AdaBoost
yy = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
wt = np.full(8, 1 / 8)
arows, alphas = [], []
for rnd, h in ((1, h1), (2, h2)):
    eps = float(wt[h != yy].sum())
    alpha = 0.5 * math.log((1 - eps) / eps)
    unnorm = wt * np.exp(-alpha * yy * h)
    after = unnorm / unnorm.sum()
    for i in range(8):
        arows.append([rnd, i + 1, wt[i], unnorm[i], after[i]])
    R[f"round{rnd}_error"], R[f"round{rnd}_alpha"] = eps, alpha
    alphas.append(alpha)
    wt = after
write_csv("outputs/part4_adaboost.csv",
          ["round", "point", "weight_before", "weight_unnormalised", "weight_after"], arows)
ens = np.sign(alphas[0] * h1 + alphas[1] * h2)
R["ensemble_accuracy"] = float(np.mean(ens == yy))

# 4.4 out-of-bag share
R["oob_fraction"] = (1 - 1 / 560) ** 560

# 4.5 architecture sizing (counts only, nothing is trained)
def out_size(inp, f, pad, stride):
    return (inp - f + 2 * pad) // stride + 1


mlp = {"mlp_hidden1": 784 * 16 + 16, "mlp_hidden2": 16 * 16 + 16, "mlp_output": 16 * 10 + 10}
c1 = out_size(28, 5, 2, 1)           # 28
p1 = out_size(c1, 2, 0, 2)           # 14
c2 = out_size(p1, 5, 0, 1)           # 10
p2 = out_size(c2, 2, 0, 2)           # 5
flat = p2 * p2 * 16                  # 400
cnn = {"cnn_conv1": 5 * 5 * 1 * 6 + 6, "cnn_conv2": 5 * 5 * 6 * 16 + 16,
       "cnn_dense1": flat * 32 + 32, "cnn_dense2": 32 * 10 + 10}
write_csv("outputs/part4_architecture.csv", ["layer", "parameters"], list({**mlp, **cnn}.items()))
R["mlp_total"], R["cnn_total"] = sum(mlp.values()), sum(cnn.values())
R["fc_layer_params"] = 128 * 128 * 3 * 1024 + 1024
R["conv_layer_params"] = 3 * 3 * 3 * 64 + 64
R["rnn_total"] = 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16
print("CNN shapes:", c1, p1, c2, p2, flat)

write_results("outputs/part4_results.txt", R)
