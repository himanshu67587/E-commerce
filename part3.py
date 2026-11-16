"""Part 3 - segmentation (K-means), hierarchical clustering, PCA. Uses all 800 rows, X1-X5."""
import numpy as np
from common import load_data, write_results, write_csv, sd

X, y, value = load_data()
n = len(y)
C = X[:, :5]
Z = (C - C.mean(axis=0)) / np.array([sd(C[:, j]) for j in range(5)])
R = {}


# 3.1 K-means with pinned starting rows
def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = None
    while True:
        d2 = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d2, axis=1)  # ties -> lower-numbered centroid
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            if np.any(labels == k):
                cent[k] = Z[labels == k].mean(axis=0)
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, cent, wcss


fits = {K: kmeans(Z, K) for K in range(2, 7)}
W = {K: fits[K][2] for K in fits}
for K in range(2, 7):
    R[f"wcss_k{K}"] = W[K]

# 3.2 elbow rule
elbow = "none"
for K in (3, 4, 5):
    if W[K] - W[K + 1] < 0.5 * (W[K - 1] - W[K]):
        elbow = K
        break
R["elbow_k"] = elbow

# 3.3 profile K=3 segments (ordered by mean X3)
labels = fits[3][0]
order = sorted(range(3), key=lambda k: C[labels == k, 2].mean())
seg_of = np.zeros(n, dtype=int)
rows = []
for name, k in zip("ABC", order):
    m = labels == k
    seg_of[m] = "ABC".index(name)
    R[f"size_segment_{name}"] = int(m.sum())
    rows.append([name, int(m.sum())] + [C[m, j].mean() for j in range(5)] + [y[m].mean(), value[m].mean()])
write_csv("outputs/part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5", "outcome_rate", "mean_value"], rows)

# 3.4 hierarchical clustering on the supplied table
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def link(a, b, rule):
    d = [D[i, j] for i in a for j in b]
    return min(d) if rule == "single" else max(d) if rule == "complete" else sum(d) / len(d)


def cname(c):
    return "+".join(f"E{i + 1}" for i in sorted(c))


hrows = []
for rule in ("single", "complete", "average"):
    clusters = [frozenset([i]) for i in range(6)]
    heights = []
    for step in range(1, 6):
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                h = link(clusters[i], clusters[j], rule)
                if best is None or h < best[0]:
                    best = (h, i, j)
        h, i, j = best
        a, b = sorted([clusters[i], clusters[j]], key=min)
        hrows.append([rule, step, cname(a), cname(b), h])
        heights.append(h)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [a | b]
    gaps = [heights[k + 1] - heights[k] for k in range(4)]
    g = int(np.argmax(gaps))          # gap between merge g+1 and g+2
    R[f"{rule}_clusters_at_cut"] = 6 - (g + 1)
write_csv("outputs/part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], hrows)

# 3.5 PCA by the seven steps
mean_vec = Z.mean(axis=0)                  # step 1 (all ~0 after standardising)
Xc = Z - mean_vec                          # step 2
S = Xc.T @ Xc / (n - 1)                    # step 3
vals, vecs = np.linalg.eigh(S)             # step 4
idx = np.argsort(vals)[::-1]               # step 5
vals, vecs = vals[idx], vecs[:, idx]
for j in range(5):                         # sign convention
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
Wm = vecs[:, :2]                           # step 6
P = Xc @ Wm                                # step 7
R["eigenvalue_sum"] = vals.sum()
for j in range(5):
    R[f"eigenvalue_{j + 1}"] = vals[j]
for j in range(5):
    R[f"explained_ratio_{j + 1}"] = vals[j] / vals.sum()
cum = np.cumsum(vals) / vals.sum()
for j in range(5):
    R[f"cumulative_ratio_{j + 1}"] = cum[j]
R["components_for_80pct"] = int(np.argmax(cum >= 0.8) + 1)
for pc in (1, 2):
    for j in range(5):
        R[f"pc{pc}_loading_X{j + 1}"] = vecs[j, pc - 1]
for r in range(3):
    R[f"row{r + 1}_pc1"], R[f"row{r + 1}_pc2"] = P[r, 0], P[r, 1]
for k, name in enumerate("ABC"):
    R[f"mean_pc1_segment_{name}"] = P[seg_of == k, 0].mean()

write_results("outputs/part3_results.txt", R)
