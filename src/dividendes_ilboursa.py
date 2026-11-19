"""Dividendes 2023-2025 non documentés par le CMF : montant ilboursa + date détectée.

Étape « fetch » : télécharge la fiche /marches/societe/<CODE> de chaque titre de l'univers
(robots.txt vérifié, pause de 3 à 5 s, cache local : une fiche n'est jamais retéléchargée).

Étape « detect » :
  - montant D par action de l'exercice N (tableau « Dividende » de la fiche) ;
    il est mis en paiement l'année N+1 ; on traite les paiements 2023, 2024 et 2025 ;
  - date de détachement : parmi les séances d'avril à août de N+1 où le Tunindex varie
    de moins de 2 %, celle dont la baisse clôture J-1 -> clôture J est la plus proche de
    D / P(J-1) (les ouvertures ilboursa ne sont pas fiables) ;
  - « détecté » seulement si l'écart ≤ 1,5 point ET une seule séance est compatible ;
  - sinon non ajusté, listé dans le rapport qualité (« non détecté » ou « ambigu »).

Sortie : data/dividendes_ilboursa.csv
  ticker, exercice, annee_paiement, montant, date_detachement, baisse_observee, baisse_attendue,
  ecart_points, nb_seances_compatibles, statut, source
"""
import argparse
import csv
import logging
import random
import re
import sys
import time
import urllib.robotparser
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_ilboursa import USER_AGENT, BASE_URL, CollectionStopped, PoliteSession  # noqa: E402
from ajustements_ilboursa import cumulative_factor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "ilboursa_societe"
EXPORTS = ROOT / "data" / "cours_corriges"  # mes_exports + data/corrections_donnees.csv (src/corriger_donnees.py)
CODES = ROOT / "data" / "codes_ilboursa.csv"
OUT = ROOT / "data" / "dividendes_ilboursa.csv"
PAY_YEARS = (2023, 2024, 2025)
TOLERANCE = 0.015
INDEX_THRESHOLD = 0.02
SOURCE = "ilboursa (montant) + détection (date)"


def universe_codes():
    with CODES.open(encoding="utf-8-sig", newline="") as f:
        return [r["code"].strip().upper() for r in csv.DictReader(f) if r.get("code", "").strip()]


def fetch(codes):
    CACHE.mkdir(parents=True, exist_ok=True)
    client = PoliteSession()
    rp = urllib.robotparser.RobotFileParser()
    rp.parse(client.request("GET", f"{BASE_URL}/robots.txt").text.splitlines())
    try:
        for code in codes:
            path = CACHE / f"{code}.html"
            if path.exists():
                continue
            url = f"{BASE_URL}/marches/societe/{code}"
            if not rp.can_fetch(USER_AGENT, url):
                raise CollectionStopped(f"robots.txt interdit {url}")
            r = client.request("GET", url)
            path.write_bytes(r.content)
            logging.info("Fiche société %s (HTTP %s)", code, r.status_code)
            print(f"fiche {code}")
    finally:
        client.close()


def to_num(s):
    s = s.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def parse_dividends(path):
    """{exercice: montant} depuis le tableau de la fiche ; {} si absent."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for table in soup.find_all("table"):
        rows = [[c.get_text(" ", strip=True) for c in r.find_all(["td", "th"])] for r in table.find_all("tr")]
        header = next((r for r in rows if r and all(re.fullmatch(r"\d{4}", x) for x in r[1:]) and len(r) > 1), None)
        div = next((r for r in rows if r and r[0].strip().lower().startswith("dividende")), None)
        if header and div:
            return {int(y): v for y, v in zip(header[1:], map(to_num, div[1:])) if v is not None and v > 0}
    return {}


def load_prices(code):
    path = EXPORTS / f"{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    if df.empty:
        return None
    out = pd.DataFrame({c: pd.to_numeric(df[c].str.replace(",", ".").str.replace(" ", ""), errors="coerce")
                        for c in ("ouverture", "cloture")})
    out.index = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return out.dropna(subset=["cloture"]).sort_index()


def confirmed_bonus_days():
    """Séances de sauts confirmés (attributions/divisions) : exclues des candidates."""
    path = ROOT / "data" / "sauts_suspects.csv"
    if not path.exists():
        return set()
    s = pd.read_csv(path, dtype=str).fillna("")
    s = s[s["confirme"].ne("non") & s["parite_retenue"].ne("")]
    return {(r["ticker"], pd.Timestamp(r["date"])) for _, r in s.iterrows()}


MIN_SHARE = 0.5  # baisse observée / baisse attendue minimale pour ajuster sans séance compatible


def detect(prices, index, amount, year, excluded=frozenset(), code=None):
    """Meilleure séance d'avril à août ; renvoie dict (vide si aucune séance)."""
    close, opens = prices["cloture"], prices["ouverture"]
    best, compatible = None, 0
    for i in range(1, len(close)):
        day = close.index[i]
        if not (pd.Timestamp(year, 4, 1) <= day <= pd.Timestamp(year, 8, 31)):
            continue
        if day in excluded:
            continue
        prev_day, prev = close.index[i - 1], close.iloc[i - 1]
        i0, i1 = index.asof(prev_day), index.asof(day)
        if pd.isna(i0) or pd.isna(i1) or abs(i1 / i0 - 1) >= INDEX_THRESHOLD:
            continue
        observed = 1 - close.iloc[i] / prev  # clôture à clôture
        # dividende par ancienne action -> unités ilboursa si l'historique a été ajusté (STAR, MPBS…)
        f = cumulative_factor(code, prev_day) if code else 1.0
        expected = amount / f / prev
        gap = abs(observed - expected)
        if gap <= TOLERANCE:
            compatible += 1
        if best is None or gap < best["ecart"]:
            best = {"date": day, "observed": observed, "expected": expected, "ecart": gap, "prev": prev, "f": f}
    if best is None:
        return {}
    best["compatible"] = compatible
    return best


def run_detect(cmf_ops_path):
    index = load_prices("PX1")["cloture"]
    cmf = pd.read_csv(cmf_ops_path, dtype=str).fillna("") if cmf_ops_path.exists() else pd.DataFrame(
        columns=["ticker", "date", "type", "document_source"])
    cmf = cmf[cmf["type"].eq("dividende") & ~cmf["document_source"].str.startswith("ilboursa")]
    # année du dividende CMF : date de paiement, à défaut date de détachement du document
    cmf_years = {(r["ticker"], (r["date"] or r.get("date_detachement_document", ""))[:4]) for _, r in cmf.iterrows()
                 if r["date"] or r.get("date_detachement_document", "")}
    bonus_days = confirmed_bonus_days()
    rows = []
    for code in universe_codes():
        excluded = {d for t, d in bonus_days if t == code}
        page = CACHE / f"{code}.html"
        divs = parse_dividends(page) if page.exists() else {}
        prices = load_prices(code)
        for year in PAY_YEARS:
            exercice = year - 1
            base = {"ticker": code, "exercice": exercice, "annee_paiement": year, "source": SOURCE}
            if (code, str(year)) in cmf_years:
                continue  # documenté par le CMF
            if not page.exists():
                rows.append({**base, "statut": "fiche société non téléchargée"})
                continue
            if exercice not in divs:
                rows.append({**base, "statut": "pas de dividende indiqué par ilboursa pour l'exercice"})
                continue
            amount = divs[exercice]
            if prices is None:
                rows.append({**base, "montant": amount, "statut": "cours indisponibles"})
                continue
            best = detect(prices, index, amount, year, excluded, code)
            if not best:
                rows.append({**base, "montant": amount, "statut": "aucune séance avril-août exploitable"})
                continue
            # Règle : une seule séance compatible -> « détecté » (date exacte) ; plusieurs -> séance
            # d'écart minimal, « approximatif » (date approximative) ; aucune -> non ajusté.
            ok = best["ecart"] <= TOLERANCE
            share = best["observed"] / best["expected"] if best["expected"] > 0 else 0
            if not ok and share >= MIN_SHARE:
                # Aucune séance compatible, mais la meilleure séance montre au moins 50 % de la baisse
                # attendue : ajustée à cette séance, date approximative.
                ok = True
                status = f"approximatif (aucune séance compatible ; baisse observée {share:.0%} de l'attendue)"
            elif not ok:
                status = f"non détecté (écart > 1,5 point ; baisse observée {share:.0%} de l'attendue) : non ajusté"
            elif best["compatible"] > 1:
                status = f"approximatif ({best['compatible']} séances compatibles)"
            else:
                status = "détecté"
            rows.append({**base, "montant": amount,
                         "date_detachement": best["date"].date().isoformat() if ok else "",
                         "meilleure_seance": best["date"].date().isoformat(),
                         "baisse_observee": round(best["observed"], 4), "baisse_attendue": round(best["expected"], 4),
                         "ecart_points": round(100 * best["ecart"], 2), "nb_seances_compatibles": best["compatible"],
                         "cours_veille": round(float(best["prev"]), 3),
                         "facteur_ilboursa": round(float(best["f"]), 6),
                         "statut": status})
    cols = ["ticker", "exercice", "annee_paiement", "montant", "date_detachement", "meilleure_seance",
            "baisse_observee", "baisse_attendue", "ecart_points", "nb_seances_compatibles", "cours_veille", "facteur_ilboursa", "statut", "source"]
    out = pd.DataFrame(rows, columns=cols)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"{len(out)} lignes -> {OUT}")
    print(out["statut"].value_counts().to_string())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("fetch", "detect", "all"), default="all")
    ap.add_argument("--codes", nargs="*", help="limiter à ces codes")
    args = ap.parse_args()
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(filename=ROOT / "logs" / "dividendes_ilboursa.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    if args.step in ("fetch", "all"):
        fetch(args.codes or universe_codes())
    if args.step in ("detect", "all"):
        run_detect(ROOT / "data" / "operations_sur_titres.csv")


if __name__ == "__main__":
    main()
