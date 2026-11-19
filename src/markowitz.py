"""Markowitz analytique : variance minimale, Sharpe maximal, frontière efficiente.

Convention : mu et Sigma annualisés (mu = moyenne des rendements log x 252).
"""
import numpy as np
from scipy.optimize import minimize


def stats(w, mu, Sigma, rf=0.0):
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ Sigma @ w))
    return {"rendement": ret, "volatilite": vol, "sharpe": (ret - rf) / vol}


def min_variance_closed_form(Sigma):
    """w* = Sigma^-1 1 / (1' Sigma^-1 1). Forme fermée, ventes à découvert autorisées."""
    x = np.linalg.solve(Sigma, np.ones(len(Sigma)))
    return x / x.sum()


def min_variance_long_only(Sigma):
    n = len(Sigma)
    res = minimize(lambda w: w @ Sigma @ w, np.ones(n) / n, jac=lambda w: 2 * Sigma @ w,
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
    return res.x


def max_sharpe_long_only(mu, Sigma, rf=0.0):
    """Reformulation convexe : min y'Sigma y  s.t.  (mu - rf)'y = 1, y >= 0 ; puis w = y / sum(y)."""
    n = len(mu)
    ex = mu - rf
    if ex.max() <= 0:
        raise ValueError("Aucun titre ne dépasse le taux sans risque : Sharpe maximal indéfini.")
    res = minimize(lambda y: y @ Sigma @ y, np.ones(n) / max(ex.sum(), 1e-9), jac=lambda y: 2 * Sigma @ y,
                   bounds=[(0, None)] * n, constraints=[{"type": "eq", "fun": lambda y: ex @ y - 1}],
                   method="SLSQP", options={"maxiter": 1000, "ftol": 1e-14})
    w = res.x / res.x.sum()
    return np.clip(w, 0, None) / np.clip(w, 0, None).sum()


def efficient_frontier(mu, Sigma, n_points=40):
    """Frontière efficiente sans vente à découvert : variance minimale pour un rendement cible."""
    n = len(mu)
    w0 = min_variance_long_only(Sigma)
    targets = np.linspace(w0 @ mu, mu.max() * 0.999, n_points)
    pts, w_init = [], w0
    for t in targets:
        res = minimize(lambda w: w @ Sigma @ w, w_init, jac=lambda w: 2 * Sigma @ w, bounds=[(0, 1)] * n,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1},
                                    {"type": "eq", "fun": lambda w, t=t: w @ mu - t}],
                       method="SLSQP", options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w_init = res.x
            pts.append((np.sqrt(res.x @ Sigma @ res.x), t))
    return np.array(pts)


def random_portfolios(mu, Sigma, rf=0.0, n=20000, seed=0):
    """Simulation aléatoire : complément pour comparaison uniquement (Dirichlet, sans vente à découvert)."""
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(len(mu)), size=n)
    ret = W @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", W, Sigma, W))
    return ret, vol, (ret - rf) / vol
