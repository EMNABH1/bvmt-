"""Préparation des données BVMT : opérations sur titres, filtres, rapport qualité.

Formats attendus (CSV, séparateur ',', dates AAAA-MM-JJ) :
  prices_long.csv        : date, ticker, close, volume
  tunindex.csv           : date, close
  corporate_actions.csv  : ticker, date, type, value
      type = split    -> value = facteur de division (1 action devient 2 : value = 2)
      type = bonus    -> value = actions gratuites pour 1 ancienne (1 pour 10 : value = 0.1)
      type = dividend -> value = dividende en dinars par action (date de détachement)
"""
import numpy as np
import pandas as pd


def load_raw(data_dir):
    prices = pd.read_csv(f"{data_dir}/prices_long.csv", parse_dates=["date"])
    index = pd.read_csv(f"{data_dir}/tunindex.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
    try:
        actions = pd.read_csv(f"{data_dir}/corporate_actions.csv", parse_dates=["date"])
    except FileNotFoundError:
        actions = pd.DataFrame(columns=["ticker", "date", "type", "value"])
    return prices, index, actions


def to_wide(prices, calendar):
    """Table dates x titres. Le calendrier est celui du Tunindex (jours de séance)."""
    close = prices.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    vol = prices.pivot_table(index="date", columns="ticker", values="volume", aggfunc="sum")
    return close.reindex(calendar), vol.reindex(calendar)


def adjust_for_actions(close, actions):
    """Ajuste les cours AVANT chaque opération (le dernier cours reste le cours réel).

    Facteur cumulé par titre : produit des facteurs des opérations postérieures à la date t.
    """
    adj = close.copy()
    log = []
    for tk, grp in actions.groupby("ticker"):
        if tk not in close.columns:
            continue
        s = close[tk]
        factor = pd.Series(1.0, index=close.index)
        for _, a in grp.sort_values("date").iterrows():
            before = s[s.index < a["date"]].dropna()
            if before.empty:
                continue
            if a["type"] == "split":
                f = 1.0 / a["value"]
            elif a["type"] == "bonus":
                f = 1.0 / (1.0 + a["value"])
            elif a["type"] == "dividend":
                f = 1.0 - a["value"] / before.iloc[-1]
            else:
                raise ValueError(f"type d'opération inconnu : {a['type']}")
            factor[factor.index < a["date"]] *= f
            log.append({"ticker": tk, "date": a["date"].date(), "type": a["type"], "value": a["value"], "facteur": round(f, 6)})
        adj[tk] = s * factor
    return adj, pd.DataFrame(log)


def flag_jumps(adj, threshold=0.10):
    """Variations journalières extrêmes restant après ajustement : à vérifier à la main."""
    r = np.log(adj).diff()
    rows = []
    for tk in r.columns:
        big = r[tk][r[tk].abs() > threshold]
        for d, v in big.items():
            rows.append({"ticker": tk, "date": d.date(), "rendement_log": round(float(v), 4)})
    return pd.DataFrame(rows)


def filter_universe(close, vol, min_history_days=500, min_trading_ratio=0.5):
    """Filtre historique minimal et liquidité. Retourne (titres retenus, tableau de statuts)."""
    rows = []
    for tk in close.columns:
        s = close[tk]
        first = s.first_valid_index()
        if first is None:
            rows.append({"ticker": tk, "retenu": False, "motif": "aucune cotation", "jours_historique": 0,
                         "jours_manquants": len(s), "ratio_jours_echanges": 0.0})
            continue
        sub = s.loc[first:]
        v = vol[tk].loc[first:].fillna(0)
        hist = len(sub)
        missing = int(sub.isna().sum())
        ratio = float((v > 0).mean())
        motif = ""
        if hist < min_history_days:
            motif = f"historique < {min_history_days} jours"
        elif ratio < min_trading_ratio:
            motif = f"jours échangés < {min_trading_ratio:.0%}"
        rows.append({"ticker": tk, "retenu": motif == "", "motif": motif, "jours_historique": hist,
                     "jours_manquants": missing, "ratio_jours_echanges": round(ratio, 3)})
    status = pd.DataFrame(rows).set_index("ticker")
    return list(status.index[status["retenu"]]), status


def prepare(data_dir, start="2023-01-01", end=None, min_history_days=500, min_trading_ratio=0.5, ffill_limit=10):
    prices, index, actions = load_raw(data_dir)
    index = index.loc[start:end]
    close, vol = to_wide(prices, index.index)
    adj, action_log = adjust_for_actions(close, actions)
    kept, status = filter_universe(adj, vol, min_history_days, min_trading_ratio)
    adj = adj[kept]
    # Cours de clôture non renseigné un jour de séance : on reporte le dernier cours (limite de jours).
    adj = adj.ffill(limit=ffill_limit)
    # Univers constant : on ne garde que les jours où tous les titres retenus ont un cours.
    adj = adj.dropna(how="any")
    index = index.reindex(adj.index)
    jumps = flag_jumps(adj)
    info = {"source": f"{data_dir}", "debut": str(adj.index[0].date()), "fin": str(adj.index[-1].date()),
            "nb_jours": len(adj), "nb_titres_depart": int(close.shape[1]), "nb_titres_retenus": len(kept)}
    return adj, index, status, action_log, jumps, info


def write_quality_report(path, status, action_log, jumps, info):
    lines = ["# Rapport qualité des données", "",
             f"- Source : {info['source']}", f"- Période : {info['debut']} au {info['fin']} ({info['nb_jours']} séances)",
             f"- Titres au départ : {info['nb_titres_depart']} ; retenus : {info['nb_titres_retenus']}", "",
             "## Jours manquants par titre", "", status[["jours_historique", "jours_manquants", "ratio_jours_echanges"]].to_markdown(), "",
             "## Titres exclus et motif", ""]
    ex = status[~status["retenu"]]
    lines.append(ex[["motif"]].to_markdown() if len(ex) else "Aucun titre exclu.")
    lines += ["", "## Ajustements effectués", "", action_log.to_markdown(index=False) if len(action_log) else "Aucun ajustement.", "",
              "## Variations extrêmes restantes (à vérifier)", "", jumps.to_markdown(index=False) if len(jumps) else "Aucune."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
