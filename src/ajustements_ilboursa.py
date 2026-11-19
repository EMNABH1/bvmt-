"""Opérations déjà ajustées rétroactivement par ilboursa dans ses cours « bruts ».

Constat : pour certains titres, les clôtures ilboursa ont plus de 2 décimales sur un bloc de séances
(ex. STAR 35.54119614655581). Ce sont des cours réels (grille au centime) divisés par un facteur F :
ilboursa a ajusté l'historique d'une attribution, d'une division ou d'une augmentation de capital.
Le cours n'a alors pas de saut le jour de l'opération, mais les dividendes publiés (fiche ilboursa,
CMF) restent par ancienne action : avant l'opération, un dividende D vaut D / F en unités ilboursa.

Méthode, par titre :
  1. séances à plus de 2 décimales, en blocs d'au moins MIN_BLOCK séances (une valeur isolée,
     comme CITY 27/04/2026 = 8.9496, est une erreur de donnée, traitée dans build_final) ;
  2. en partant de la fin du bloc, recherche du facteur f (grille 1 à 6, pas 5e-6) qui ramène les
     cours sur la grille du centime ; le segment s'étend tant que les cours restent sur la grille ;
     les séances antérieures forment un nouveau segment (opération plus ancienne, facteur cumulé) ;
  3. facteur affiné (médiane de arrondi(P·f, 2) / P), remplacé par une fraction simple si l'écart
     relatif est < 2e-5 ;
  4. facteur de chaque opération = rapport des facteurs cumulés de deux segments successifs ;
     date d'effet = première séance après le segment.

Sortie : data/ajustements_ilboursa.csv. La fonction cumulative_factor(ticker, jour) donne F(jour).
"""
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "mes_exports"
OUT = ROOT / "data" / "ajustements_ilboursa.csv"
MIN_BLOCK = 20
TOL = 0.011  # en centimes
GRID = np.arange(1.0, 6.0, 0.000005)


def load_raw(code):
    df = pd.read_csv(EXPORTS / f"{code}.csv", sep=";", encoding="utf-8-sig", dtype=str)
    if df.empty:
        return pd.Series(dtype=str)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df.set_index("date").sort_index()["cloture"].str.replace(",", ".").str.strip()


def on_grid(values, f):
    x = values * f * 100
    return np.abs(x - np.round(x)) < TOL


def fit(values):
    best, score = None, -1.0
    for i in range(0, len(GRID), 20000):
        g = GRID[i:i + 20000][:, None]
        x = values[None, :] * g * 100
        sc = (np.abs(x - np.round(x)) < TOL).mean(axis=1)
        j = int(sc.argmax())
        if sc[j] > score:
            score, best = float(sc[j]), float(g[j, 0])
    return best, score


def refine(values, f):
    f = float(np.median(np.round(values * f, 2) / values))
    fr = Fraction(f).limit_denominator(60)
    return (float(fr), f"{fr.numerator}/{fr.denominator}") if abs(float(fr) / f - 1) < 2e-5 else (f, f"{f:.6f}")


def segments(code):
    """Liste de (debut, fin, facteur_cumule, libelle, score) du plus récent au plus ancien."""
    s = load_raw(code)
    if s.empty:
        return []
    dec = s.str.split(".").str[1].fillna("").str.len()
    adj = s[dec > 2].astype(float)
    if len(adj) < MIN_BLOCK:
        return []
    out = []
    rest = adj
    while len(rest) >= MIN_BLOCK:
        f, _ = fit(rest.values[-min(80, len(rest)):])
        ok = pd.Series(on_grid(rest.values, f), index=rest.index)
        bad = ok[~ok].index
        # le segment récent s'étend jusqu'à la dernière séance hors grille (tolère 5 % d'exceptions)
        start = rest.index.min()
        if len(bad):
            for b in sorted(bad, reverse=True):
                tail = ok[ok.index > b]
                if len(tail) >= MIN_BLOCK and tail.mean() >= 0.95 and ok[ok.index <= b].tail(10).mean() < 0.5:
                    start = tail.index.min()
                    break
        seg = rest[rest.index >= start]
        fval, label = refine(seg.values, f)
        out.append((seg.index.min(), seg.index.max(), fval, label, round(float(on_grid(seg.values, fval).mean()), 3)))
        rest = rest[rest.index < start]
    return out


def build(codes):
    rows = []
    for code in codes:
        segs = segments(code)
        if not segs:
            continue
        raw = load_raw(code)
        prev_cum = 1.0
        for debut, fin, cum, label, score in segs:  # du plus récent au plus ancien
            eff = raw.index[raw.index > fin]
            factor = cum / prev_cum
            fr = Fraction(factor).limit_denominator(60)
            flabel = f"{fr.numerator}/{fr.denominator}" if abs(float(fr) / factor - 1) < 2e-5 else f"{factor:.6f}"
            rows.append({"ticker": code, "debut_bloc": debut.date(), "fin_bloc": fin.date(),
                         "date_effet": eff[0].date() if len(eff) else "", "facteur_operation": round(factor, 6),
                         "facteur_operation_libelle": flabel, "facteur_cumule": round(cum, 6),
                         "facteur_cumule_libelle": label, "part_seances_sur_grille": score})
            prev_cum = cum
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False, encoding="utf-8")
    return df


@lru_cache(maxsize=None)
def _table():
    if not OUT.exists():
        return {}
    df = pd.read_csv(OUT, dtype={"ticker": str})
    t = {}
    for _, r in df.iterrows():
        t.setdefault(r["ticker"], []).append((pd.Timestamp(r["debut_bloc"]), pd.Timestamp(r["fin_bloc"]),
                                              float(r["facteur_cumule"])))
    return t


def cumulative_factor(ticker, day):
    """F(jour) : diviseur appliqué par ilboursa aux cours de ce jour (1 hors des blocs ajustés)."""
    day = pd.Timestamp(day)
    for debut, fin, cum in _table().get(ticker, []):
        if debut <= day <= fin:
            return cum
    return 1.0


if __name__ == "__main__":
    codes = sorted(p.stem for p in EXPORTS.glob("*.csv") if p.stem != "PX1")
    df = build(codes)
    print(df.to_string(index=False) if len(df) else "aucun bloc ajusté")
    print(f"-> {OUT}")
