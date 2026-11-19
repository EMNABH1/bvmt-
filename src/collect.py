"""Collecte des historiques BVMT (reproductible, respectueuse des sites).

IMPORTANT : ce script n'a pas pu être testé contre les sites réels (pas d'accès réseau lors de sa rédaction).
Avant usage :
  1. Ouvrez la page d'historique d'une valeur sur le site choisi (BVMT ou ilboursa) et lisez son robots.txt
     et ses conditions d'utilisation. Faites valider toute autre source par l'encadrant (semaine 1).
  2. Renseignez URL_TEMPLATE ci-dessous (avec {ticker}) d'après l'URL réellement observée.
  3. Vérifiez que parse_history() retrouve bien les colonnes date / clôture / volume (noms d'en-têtes à adapter).
Alternative sans scraping : exportez les historiques (fichiers fournis par la BVMT, service « données
historiques ») puis utilisez  python src/collect.py --from-csv-dir dossier_csv  pour les normaliser.

Format de sortie : prices_long.csv (date, ticker, close, volume) et tunindex.csv (date, close).
"""
import argparse
import glob
import io
import os
import time
import urllib.robotparser
from urllib.parse import urlparse

import pandas as pd
import requests

USER_AGENT = "ESPRIT-5INFINI-student-project (contact: votre.email@esprit.tn)"
URL_TEMPLATE = "https://EXEMPLE.tn/historique?valeur={ticker}"   # À REMPLIR
DELAY_SECONDS = 3.0                                               # requêtes espacées

DATE_NAMES = ("date", "séance", "seance")
CLOSE_NAMES = ("clôture", "cloture", "dernier", "close", "cours")
VOLUME_NAMES = ("volume", "quantité", "quantite", "titres")


def allowed(url):
    p = urlparse(url)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(f"{p.scheme}://{p.netloc}/robots.txt")
    rp.read()
    return rp.can_fetch(USER_AGENT, url)


def _pick(columns, names):
    for c in columns:
        if any(n in str(c).lower() for n in names):
            return c
    raise KeyError(f"colonne introuvable parmi {list(columns)} (cherchait {names})")


def normalize_table(df):
    df = df.copy()
    d, c, v = _pick(df.columns, DATE_NAMES), _pick(df.columns, CLOSE_NAMES), _pick(df.columns, VOLUME_NAMES)
    out = pd.DataFrame({"date": pd.to_datetime(df[d], dayfirst=True, errors="coerce"),
                        "close": pd.to_numeric(df[c].astype(str).str.replace(r"\s", "", regex=True).str.replace(",", "."), errors="coerce"),
                        "volume": pd.to_numeric(df[v].astype(str).str.replace(r"\s", "", regex=True).str.replace(",", "."), errors="coerce")})
    return out.dropna(subset=["date", "close"]).sort_values("date")


def parse_history(html):
    for t in pd.read_html(io.StringIO(html)):
        try:
            return normalize_table(t)
        except KeyError:
            continue
    raise ValueError("aucun tableau d'historique reconnu")


def scrape(tickers, out_dir, url_template=URL_TEMPLATE):
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for tk in tickers:
        url = url_template.format(ticker=tk)
        if not allowed(url):
            print(f"robots.txt interdit {url} : titre ignoré")
            continue
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        r.raise_for_status()
        with open(f"{out_dir}/raw_{tk}.html", "w", encoding="utf-8") as f:    # trace brute pour la reproductibilité
            f.write(r.text)
        h = parse_history(r.text)
        h["ticker"] = tk
        frames.append(h)
        time.sleep(DELAY_SECONDS)
    pd.concat(frames)[["date", "ticker", "close", "volume"]].to_csv(f"{out_dir}/prices_long.csv", index=False)


def from_csv_dir(csv_dir, out_dir):
    """Un fichier CSV par titre (nom du fichier = code du titre) ; 'TUNINDEX.csv' pour l'indice."""
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for path in sorted(glob.glob(f"{csv_dir}/*.csv")):
        name = os.path.splitext(os.path.basename(path))[0]
        raw = pd.read_csv(path, sep=None, engine="python")
        n = normalize_table(raw)
        if name.upper() == "TUNINDEX":
            n[["date", "close"]].to_csv(f"{out_dir}/tunindex.csv", index=False)
        else:
            n["ticker"] = name
            frames.append(n[["date", "ticker", "close", "volume"]])
    pd.concat(frames).to_csv(f"{out_dir}/prices_long.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv-dir")
    ap.add_argument("--tickers-file", help="CSV avec une colonne 'ticker'")
    ap.add_argument("--out-dir", default="data/raw")
    a = ap.parse_args()
    if a.from_csv_dir:
        from_csv_dir(a.from_csv_dir, a.out_dir)
    else:
        scrape(pd.read_csv(a.tickers_file)["ticker"].tolist(), a.out_dir)
