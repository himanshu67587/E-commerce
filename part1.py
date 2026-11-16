"""Part 1 - estimation, inference and regression baseline (all 800 rows)."""
import math
import numpy as np
from common import load_data, write_results, sd

X, y_out, value = load_data()
n = len(value)
R = {}

# 1.1 estimate with an interval
mean_value = value.mean()
sd_value = sd(value)
se_value = sd_value / math.sqrt(n)
R.update(n=n, mean_value=mean_value, sd_value=sd_value, se_value=se_value,
         ci95_low=mean_value - 1.96 * se_value, ci95_high=mean_value + 1.96 * se_value)

# 1.2 test H0: mu = 8.0
z_stat = (mean_value - 8.0) / se_value
R.update(z_stat=z_stat, reject_h0=abs(z_stat) > 1.96)

# 1.3 required sample size
R["n_required"] = math.ceil((1.96 * sd_value / 0.5) ** 2)

# 1.4 simple regression, closed form
x4 = X[:, 3]


def simple_ols(x, y):
    xm, ym = x.mean(), y.mean()
    slope = (np.sum((x - xm) * (y - ym)) / (len(x) - 1)) / (np.sum((x - xm) ** 2) / (len(x) - 1))
    return ym - slope * xm, slope


icpt, slope = simple_ols(x4, value)
fitted = icpt + slope * x4
resid = value - fitted
tss = np.sum((value - value.mean()) ** 2)
ess = np.sum((fitted - value.mean()) ** 2)
rss = np.sum(resid ** 2)
R.update(reg_slope=slope, reg_intercept=icpt, tss=tss, ess=ess, rss=rss,
         tss_equals_ess_plus_rss=abs(tss - ess - rss) < 1e-6 * tss,
         r2=1 - rss / tss, rmse=math.sqrt(rss / n), mae=np.mean(np.abs(resid)))

# 1.5 gradient descent on standardised X4
mx, sx = x4.mean(), sd(x4)
z = (x4 - mx) / sx
m = c = 0.0
cost = lambda m, c: float(np.mean((value - (m * z + c)) ** 2))
R["gd_J0"] = cost(m, c)
for step in range(1, 501):
    e = value - (m * z + c)
    gm, gc = -2.0 / n * np.sum(z * e), -2.0 / n * np.sum(e)
    m, c = m - 0.1 * gm, c - 0.1 * gc
    if step in (1, 10, 100, 500):
        R[f"gd_J{step}"] = cost(m, c)
R["gd_slope"] = m / sx
R["gd_intercept"] = c - m * mx / sx
R["gd_slope_gap"] = abs(R["gd_slope"] - slope)

# 1.6 funnel check and log remedy
def halves_sd(fit, res):
    order = np.argsort(fit, kind="stable")
    return sd(res[order[:400]]), sd(res[order[400:]])


lo, up = halves_sd(fitted, resid)
R.update(resid_sd_lower_raw=lo, resid_sd_upper_raw=up, resid_sd_ratio_raw=up / lo)
ly, lx = np.log(value), np.log(x4)
l_icpt, l_slope = simple_ols(lx, ly)
l_fit = l_icpt + l_slope * lx
l_res = ly - l_fit
l_tss, l_rss = np.sum((ly - ly.mean()) ** 2), np.sum(l_res ** 2)
l_r2 = 1 - l_rss / l_tss
lo, up = halves_sd(l_fit, l_res)
R.update(loglog_intercept=l_icpt, loglog_slope=l_slope, loglog_r2=l_r2,
         loglog_rmse=math.sqrt(l_rss / n), resid_sd_lower_log=lo, resid_sd_upper_log=up,
         resid_sd_ratio_log=up / lo)

# 1.7 multiple regression, normal equation
D = np.column_stack([np.ones(n), lx, X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(D.T @ D, D.T @ ly)
res = ly - D @ beta
r2 = 1 - np.sum(res ** 2) / l_tss
adj = lambda r2, p: 1 - (1 - r2) * (n - 1) / (n - p - 1)
for i, b in enumerate(beta):
    R[f"mr_beta_{i}"] = b
R.update(mr_r2=r2, mr_adj_r2=adj(r2, 7), loglog_adj_r2=adj(l_r2, 1))

write_results("outputs/part1_results.txt", R)
