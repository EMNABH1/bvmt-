"""Sauts non confirmés mais très suspects : recherche d'un avis, sinon liste pour vérification.

Sélection dans data/sauts_suspects.csv (produit par detect_sauts.py) :
  - confirme = « non » ;
  - baisse clôture à clôture ≥ 8 % ;
  - à moins de 1 % d'une parité réaliste (proche_parite_simple = oui) ;
  - Tunindex < 2 % en valeur absolue (déjà garanti par la détection) ;
  - pas de petit cours (P < 1 DT : rapports « exacts » artificiels, voir detect_sauts.py) ;
  - pas de rebond le lendemain : la séance cotée suivante reprend moins de la moitié
    de la baisse (P(t+1) < P(t) + 0,5 × (P(t-1) − P(t))).

Pour chaque saut retenu :
  1. avis parmi les « augmentations de capital réalisées » du CMF (index_augmentations.csv
     et textes des PDF) : même titre, publié entre 120 jours avant et 60 jours après ;
  2. actualités ilboursa du titre (news_valeur, titres et dates de la liste) : titres
     contenant gratuit / attribution / augmentation de capital / division / nominal /
     incorporation de réserves, publiés entre 120 jours avant et 30 jours après.
     Pages téléchargées poliment (robots.txt, pause 3-5 s) et mises en cache.

Aucun ajustement n'est fait ici. Sortie : data/sauts_a_verifier.csv, reprise telle quelle
dans le rapport qualité (statut « aucun avis : non ajusté » ou « avis trouvé : à valider »).
"""
import argparse
import re
import sys
import urllib.robotparser
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_ilboursa import BASE_URL, USER_AGENT, CollectionStopped, PoliteSession  # noqa: E402
from univers import fold, ticker_in_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAUTS = ROOT / "data" / "sauts_suspects.csv"
EXPORTS = ROOT / "data" / "cours_corriges"  # mes_exports + data/corrections_donnees.csv (src/corriger_donnees.py)
CMF = ROOT / "data" / "ost_cmf"
NEWS_CACHE = ROOT / "data" / "ilboursa_actus"
OUT = ROOT / "data" / "sauts_a_verifier.csv"
MIN_DROP = 0.08
KEYWORDS = ("gratuit", "attribution", "augmentation de capital", "division", "nominal",
            "incorporation de reserves", "split")
MAX_NEWS_PAGES = 40


def candidates():
    s = pd.read_csv(SAUTS, dtype={"ticker": str})
    s = s[s["confirme"].eq("non") & (s["variation_cloture"] <= -MIN_DROP)
          & s["proche_parite_simple"].eq("oui") & s["petit_cours"].eq("non")
          & (s["variation_tunindex"].abs() < 0.02)].copy()
    keep = []
    for _, r in s.iterrows():
        df = pd.read_csv(EXPORTS / f"{r['ticker']}.csv", sep=";", encoding="utf-8-sig", dtype=str)
        close = pd.to_numeric(df["cloture"].str.replace(",", "."), errors="coerce")
        close.index = pd.to_datetime(df["date"], format="%d/%m/%Y")
        close = close.dropna().sort_index()
        after = close[close.index > pd.Timestamp(r["date"])]
        if after.empty:
            nxt, rebound = None, None
        else:
            nxt = after.iloc[0]
            rebound = (nxt - r["cours"]) / (r["cours_precedent"] - r["cours"])
        if rebound is not None and rebound >= 0.5:
            continue
        r = r.copy()
        r["cours_lendemain"] = nxt
        r["part_baisse_reprise_lendemain"] = None if rebound is None else round(rebound, 2)
        keep.append(r)
    return pd.DataFrame(keep)


def cmf_capital_notices(ticker, day):
    path = CMF / "index_augmentations.csv"
    if not path.exists():
        return []
    aug = pd.read_csv(path, dtype=str).fillna("")
    found = []
    for _, a in aug.iterrows():
        if ticker_in_text(a["societe"]) != ticker or not a["date_avis"]:
            continue
        d = pd.Timestamp(a["date_avis"])
        if not (day - timedelta(days=120) <= d <= day + timedelta(days=60)):
            continue
        txt = CMF / "txt" / "augmentations" / (Path(a["url"]).stem + ".txt")
        nature = ""
        if txt.exists():
            t = fold(txt.read_text(encoding="utf-8", errors="replace"))
            nature = "gratuite/incorporation" if ("gratuit" in t or "incorporation" in t) else "en numéraire ou autre"
        found.append(f"{a['date_avis']} {a['societe']} ({nature or 'texte absent'}) <{a['url']}>")
    return found


class News:
    def __init__(self, fetch):
        self.fetch = fetch
        self.client = None
        self.rp = None

    def page(self, code, n):
        NEWS_CACHE.mkdir(parents=True, exist_ok=True)
        path = NEWS_CACHE / f"{code}_p{n}.html"
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        if not self.fetch:
            return None
        if self.client is None:
            self.client = PoliteSession()
            self.rp = urllib.robotparser.RobotFileParser()
            self.rp.parse(self.client.request("GET", f"{BASE_URL}/robots.txt").text.splitlines())
        url = f"{BASE_URL}/marches/news_valeur?s={code}" + (f"&p={n}" if n > 1 else "")
        if not self.rp.can_fetch(USER_AGENT, url):
            raise CollectionStopped(f"robots.txt interdit {url}")
        html = self.client.request("GET", url).text
        path.write_text(html, encoding="utf-8")
        return html

    def items(self, code, oldest):
        """(date, titre, url) des actualités du titre jusqu'à la date « oldest »."""
        out = []
        for n in range(1, MAX_NEWS_PAGES + 1):
            html = self.page(code, n)
            if html is None:
                break
            soup = BeautifulSoup(html, "html.parser")
            page_items = []
            for span in soup.select("span.sp1"):
                a = span.find_previous("a", href=True)
                m = re.match(r"(\d{2})/(\d{2})/(\d{2})", span.get_text(strip=True))
                if a is None or not m:
                    continue
                d = datetime(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)))
                page_items.append((pd.Timestamp(d), a.get_text(" ", strip=True), f"{BASE_URL}/marches/{a['href']}"))
            if not page_items:
                break
            out += page_items
            if min(d for d, _, _ in page_items) < oldest:
                break
        return out

    def close(self):
        if self.client:
            self.client.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="n'utiliser que le cache des actualités")
    args = ap.parse_args()
    cand = candidates()
    print(f"{len(cand)} sauts non confirmés très suspects")
    news = News(fetch=not args.no_fetch)
    rows = []
    try:
        for code, grp in cand.groupby("ticker"):
            oldest = pd.Timestamp(grp["date"].min()) - timedelta(days=120)
            items = news.items(code, oldest)
            for _, r in grp.iterrows():
                day = pd.Timestamp(r["date"])
                cmf = cmf_capital_notices(code, day)
                actus = [f"{d.date()} {t} <{u}>" for d, t, u in items
                         if day - timedelta(days=120) <= d <= day + timedelta(days=30)
                         and any(k in fold(t) for k in KEYWORDS)]
                rows.append({
                    "ticker": code, "date": r["date"], "cours_precedent": r["cours_precedent"], "cours": r["cours"],
                    "cours_lendemain": r["cours_lendemain"], "variation_cloture": r["variation_cloture"],
                    "variation_tunindex": r["variation_tunindex"], "rapport_P_prec_sur_P": r["rapport_P_prec_sur_P"],
                    "parite_la_plus_proche": r["parite_la_plus_proche"], "libelle_parite": r["libelle_parite"],
                    "part_baisse_reprise_lendemain": r["part_baisse_reprise_lendemain"],
                    "actus_ilboursa_parcourues": "oui" if items else "non (non téléchargées)",
                    "avis_cmf_augmentation": " || ".join(cmf), "actus_ilboursa": " || ".join(actus),
                    "statut": "avis trouvé : à valider" if (cmf or actus) else "aucun avis : non ajusté",
                })
    finally:
        news.close()
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"-> {OUT}")
    if len(out):
        print(out[["ticker", "date", "variation_cloture", "parite_la_plus_proche", "part_baisse_reprise_lendemain",
                   "statut"]].to_string(index=False))


if __name__ == "__main__":
    main()
