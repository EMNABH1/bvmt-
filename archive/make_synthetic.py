"""Jeu de données SYNTHÉTIQUE, uniquement pour tester le pipeline. Ce ne sont PAS des données de la BVMT.

Il reproduit le format attendu et inclut volontairement : un titre introduit trop récemment, un titre
illiquide, une division d'action, des dividendes et un titre atypique.
"""
import sys
import numpy as np
import pandas as pd


def make(out_dir, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-02", "2026-06-30")
    T, n = len(dates), 48
    rm = rng.normal(0.0004, 0.006, T)                                 # marché
    groups = rng.integers(0, 3, n)                                    # 3 profils
    base_vol = np.array([0.008, 0.013, 0.02])[groups]
    base_mu = np.array([0.0003, 0.0005, 0.0009])[groups]
    beta = np.array([0.5, 0.9, 1.3])[groups] + rng.normal(0, 0.1, n)
    rets = rm[:, None] * beta + rng.normal(base_mu, base_vol, (T, n)) * 0.8
    rets[:, 0] = rng.standard_t(2.5, T) * 0.03                       # titre atypique
    tickers = [f"TN{i:03d}" for i in range(n)]
    px = 20 * np.exp(np.cumsum(rets, axis=0))
    vol = rng.integers(500, 20000, (T, n)).astype(float)
    # titre illiquide (peu d'échanges) et titre introduit récemment
    vol[rng.random(T) < 0.8, 1] = 0
    px_df = pd.DataFrame(px, index=dates, columns=tickers)
    px_df.iloc[:-250, 2] = np.nan
    vol_df = pd.DataFrame(vol, index=dates, columns=tickers)
    # division d'action 1 -> 2 pour TN003 : cours brut divisé par 2 à la date de division
    split_date = dates[600]
    px_df.loc[px_df.index >= split_date, "TN003"] /= 2.0
    actions = [{"ticker": "TN003", "date": split_date.date(), "type": "split", "value": 2.0}]
    for tk, d in [("TN004", dates[300]), ("TN005", dates[560])]:
        before = px_df[tk][px_df.index < d].iloc[-1]
        div = round(0.03 * before, 3)
        px_df.loc[px_df.index >= d, tk] *= (1 - div / before)         # détachement : cours brut baisse
        actions.append({"ticker": tk, "date": d.date(), "type": "dividend", "value": div})
    long = px_df.stack().rename("close").reset_index()
    long.columns = ["date", "ticker", "close"]
    long["volume"] = [vol_df.at[d, t] for d, t in zip(long["date"], long["ticker"])]
    long.to_csv(f"{out_dir}/prices_long.csv", index=False)
    idx = 1000 * np.exp(np.cumsum(rm))
    pd.DataFrame({"date": dates, "close": idx}).to_csv(f"{out_dir}/tunindex.csv", index=False)
    pd.DataFrame(actions).to_csv(f"{out_dir}/corporate_actions.csv", index=False)


if __name__ == "__main__":
    make(sys.argv[1])
