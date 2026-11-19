"""Corrections des erreurs de données ilboursa, appliquées une fois pour toutes les étapes.

Entrées  : mes_exports/<CODE>.csv (cours bruts ilboursa, jamais modifiés)
           data/corrections_donnees.csv (ticker, date AAAA-MM-JJ, action, motif)
Sorties  : data/cours_corriges/<CODE>.csv (même format, séances erronées supprimées)
           data/prix_decimales_anormales.csv : toutes les clôtures à plus de 2 décimales, classées
             « bloc ajusté par ilboursa » (voir src/ajustements_ilboursa.py) ou « valeur isolée ».

Action reconnue : « supprimer » (la séance est retirée ; le titre est traité comme non coté ce jour).
Les étapes suivantes (extract_ost, detect_sauts, dividendes_ilboursa, verifier_sauts, build_final)
lisent data/cours_corriges.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "mes_exports"
OUT = ROOT / "data" / "cours_corriges"
CORRECTIONS = ROOT / "data" / "corrections_donnees.csv"
SCAN = ROOT / "data" / "prix_decimales_anormales.csv"
MIN_BLOCK = 20


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corr = pd.read_csv(CORRECTIONS, dtype=str) if CORRECTIONS.exists() else pd.DataFrame(
        columns=["ticker", "date", "action", "motif"])
    applied, scan = [], []
    for path in sorted(RAW.glob("*.csv")):
        code = path.stem
        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
        if len(df):
            dates = pd.to_datetime(df["date"], format="%d/%m/%Y")
            close = df["cloture"].str.replace(",", ".").str.strip()
            dec = close.str.split(".").str[1].fillna("").str.len()
            n_block = int((dec > 2).sum())
            for i in df.index[dec > 2]:
                scan.append({"ticker": code, "date": dates[i].date(), "cloture": close[i],
                             "classement": "bloc ajusté par ilboursa" if n_block >= MIN_BLOCK else "valeur isolée"})
            drop = corr[(corr["ticker"] == code) & (corr["action"] == "supprimer")]
            for _, c in drop.iterrows():
                hit = dates.dt.strftime("%Y-%m-%d") == c["date"]
                if hit.any():
                    applied.append({**c.to_dict(), "cloture_supprimee": close[hit].iloc[0]})
                    df = df[~hit.values]
                    dates = dates[~hit.values]
                    close = close[~hit.values]
        df.to_csv(OUT / path.name, sep=";", index=False, encoding="utf-8-sig")
    pd.DataFrame(scan).to_csv(SCAN, index=False, encoding="utf-8")
    pd.DataFrame(applied).to_csv(ROOT / "data" / "corrections_appliquees.csv", index=False, encoding="utf-8")
    sc = pd.DataFrame(scan)
    print(f"{len(applied)} correction(s) appliquée(s) -> {OUT}")
    if len(sc):
        print(sc.groupby(["ticker", "classement"]).size().to_string())


if __name__ == "__main__":
    main()
