"""Features par titre, calculées sur les rendements logarithmiques."""
import numpy as np
import pandas as pd
from scipy import stats

ANN = 252  # jours de séance par an (hypothèse à documenter)
FEATURES = ["rendement_ann", "volatilite_ann", "skewness", "kurtosis", "VaR95", "CVaR95", "beta"]


def log_returns(prices):
    return np.log(prices).diff().dropna(how="all")


def compute_features(prices, index_close, alpha=0.05):
    r = log_returns(prices)
    rm = np.log(index_close).diff().reindex(r.index)
    out = pd.DataFrame(index=r.columns)
    out["rendement_ann"] = r.mean() * ANN
    out["volatilite_ann"] = r.std(ddof=1) * np.sqrt(ANN)
    out["skewness"] = r.apply(lambda x: stats.skew(x, bias=False))
    out["kurtosis"] = r.apply(lambda x: stats.kurtosis(x, fisher=True, bias=False))  # excès de kurtosis
    q = r.quantile(alpha)
    out["VaR95"] = -q                                        # perte journalière positive
    out["CVaR95"] = pd.Series({c: -r[c][r[c] <= q[c]].mean() for c in r.columns})
    out["beta"] = r.apply(lambda x: np.cov(x, rm)[0, 1] / np.var(rm, ddof=1))
    return out[FEATURES], r


def feature_correlations(features):
    return features.corr(method="pearson"), features.corr(method="spearman")
