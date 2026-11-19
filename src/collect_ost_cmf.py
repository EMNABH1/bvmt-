"""Collecte des documents d'opérations sur titres publiés par le CMF (cmf.tn).

Deux rubriques publiques :
  - « Résolutions adoptées par les AG des sociétés faisant APE » : dividende par action,
    date de mise en paiement, attributions d'actions gratuites décidées.
  - « Augmentations de capital réalisées » : avis de réalisation (parité, date).

Respect du site : robots.txt vérifié, Crawl-delay de 10 s respecté, cache local
(aucun document n'est retéléchargé), arrêt immédiat sur 403/429/503.

Sorties (data/ost_cmf/) :
  index_resolutions.csv, index_augmentations.csv, pdf/*.pdf, txt/*.txt
"""
import argparse
import csv
import logging
import re
import shutil
import subprocess
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

try:
    import pdfplumber
except ImportError:  # pip install pdfplumber (requirements.txt)
    pdfplumber = None

BASE = "https://www.cmf.tn"
USER_AGENT = "BVMT-Phase2-student-project/1.0 (academic data collection)"
CRAWL_DELAY = 10.5
RESOLUTIONS = "/?q=r-solutions-adopt-es-par-les-assembl-s-g-n-rales-des-soci-t-s-faisant-ape"
AUGMENTATIONS = "/?q=augmentations-de-capital-r-alis-es"
INFOS = "/?q=informations-des-soci-t-s"
NL = chr(10)
INFO_KEYWORDS = ("dividende", "detachement", "mise en paiement", "attribution", "actions gratuites",
                 "augmentation de capital", "division", "nominal")
MONTHS = {"jan": 1, "fév": 2, "fev": 2, "mar": 3, "avr": 4, "mai": 5, "juin": 6, "juil": 7,
          "aoû": 8, "aou": 8, "sep": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12}
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "ost_cmf"
YEARS = {"2023", "2024", "2025", "2026"}


class Stop(RuntimeError):
    pass


class Client:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.last = 0.0
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(self.get(BASE + "/robots.txt").text.splitlines())
        self.rp = rp

    def get(self, url):
        wait = self.last + CRAWL_DELAY - time.time()
        if wait > 0:
            time.sleep(wait)
        if getattr(self, "rp", None) and not self.rp.can_fetch(USER_AGENT, url):
            raise Stop(f"robots.txt interdit {url}")
        r = self.s.get(url, timeout=60)
        self.last = time.time()
        if r.status_code in (403, 429, 503):
            raise Stop(f"HTTP {r.status_code} sur {url}")
        r.raise_for_status()
        logging.info("GET %s -> %s (%d o)", url, r.status_code, len(r.content))
        return r


def parse_fr_date(text):
    """« 9 oct 2026 », « 1er sep 2026 », « 15 décembre 2025 » -> AAAA-MM-JJ ; '' sinon (journalisé)."""
    m = re.search(r"(\d{1,2})\s*(?:er|ère)?\s+([a-zéèûô]+)\.?\s+(\d{4})", text.lower())
    mon = None
    if m:
        word = m.group(2)
        mon = MONTHS.get(word[:4]) or MONTHS.get(word[:3])
    if not mon:
        logging.warning("Date non reconnue : %r", text)
        return ""
    return f"{m.group(3)}-{mon:02d}-{int(m.group(1)):02d}"


def crawl_resolutions(client, max_pages):
    rows = []
    for page in range(max_pages):
        html = client.get(f"{BASE}{RESOLUTIONS}&page={page}").text
        soup = BeautifulSoup(html, "html.parser")
        view = soup.select_one(".view-content")
        if view is None:
            break
        page_rows = view.select(".views-row")
        years_seen = set()
        for row in page_rows:
            parts = [p.strip() for p in row.get_text("|", strip=True).split("|") if p.strip()]
            links = list(dict.fromkeys(a["href"] for a in row.find_all("a", href=True) if a["href"].lower().endswith(".pdf")))
            year = next((p for p in parts if re.fullmatch(r"\d{4}", p)), "")
            years_seen.add(year)
            for link in links:
                rows.append({"societe": parts[0] if parts else "", "categorie": parts[1] if len(parts) > 1 else "",
                             "annee": year, "url": link})
        print(f"résolutions page {page}: {len(page_rows)} lignes, années {sorted(years_seen)}")
        if soup.select_one(".pager-next") is None:
            break
    return rows


def crawl_augmentations(client, max_pages):
    rows = []
    for page in range(max_pages):
        soup = BeautifulSoup(client.get(f"{BASE}{AUGMENTATIONS}&page={page}").text, "html.parser")
        view = soup.select_one(".view-content")
        if view is None:
            break
        dates = []
        for row in view.select(".views-row"):
            parts = [p.strip() for p in row.get_text("|", strip=True).split("|") if p.strip()]
            a = row.find("a", href=True)
            d = parse_fr_date(parts[0]) if parts else ""
            dates.append(d)
            if d and d[:4] in YEARS and a:
                rows.append({"date_avis": d, "societe": parts[-1], "page": urljoin(BASE, a["href"]), "url": ""})
        print(f"augmentations page {page}: dates {min(filter(None, dates), default='')}..{max(filter(None, dates), default='')}")
        if dates and all(d and d[:4] < min(YEARS) for d in dates):
            break
    for row in rows:
        soup = BeautifulSoup(client.get(row["page"]).text, "html.parser")
        pdfs = [a["href"] for a in soup.find_all("a", href=True) if a["href"].lower().endswith(".pdf")]
        row["url"] = urljoin(BASE, pdfs[0]) if pdfs else ""
    return rows


def crawl_infos(client, max_pages, start_page=0):
    """Index « Informations des sociétés » de 2023 à aujourd'hui.

    Garde les communiqués dont le titre contient un mot-clé d'opération sur titres et
    qui concernent une société de l'univers. Les autres titres à mot-clé sont listés
    à part (index_infos_hors_univers.csv) pour contrôle.
    """
    from univers import fold, ticker_in_text
    kept, other = [], []
    for page in range(start_page, max_pages):
        soup = BeautifulSoup(client.get(f"{BASE}{INFOS}&page={page}").text, "html.parser")
        rows = soup.select(".view-content .views-row")
        if not rows:
            break
        dates = []
        for row in rows:
            parts = [x.strip() for x in row.get_text("|", strip=True).split("|") if x.strip()]
            m = re.match(r"(\d{2})/(\d{2})/(\d{4})", parts[0] if parts else "")
            if not m:
                logging.warning("Date non reconnue (infos) : %r", parts[:1])
                continue
            d = f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
            dates.append(d)
            title = " ".join(parts[1:])
            a = row.find("a", href=True)
            if d < "2023-01-01" or a is None or not any(k in fold(title) for k in INFO_KEYWORDS):
                continue
            # Société : d'abord après le dernier « : » du titre, sinon dans tout le titre.
            company = title.rsplit(":", 1)[1] if ":" in title else ""
            ticker = ticker_in_text(company) if company else None
            ticker = ticker or ticker_in_text(title)
            rec = {"date_publication": d, "ticker": ticker or "", "titre": title, "page": urljoin(BASE, a["href"])}
            (kept if ticker else other).append(rec)
        print(f"infos page {page}: {min(dates, default='')}..{max(dates, default='')}, retenus cumulés {len(kept)}")
        if dates and max(dates) < "2023-01-01":
            break
    return kept, other


def fetch_infos(client, rows):
    """Télécharge chaque communiqué retenu : texte HTML de la page, plus ses PDF éventuels."""
    html_dir, pdf_dir, txt_dir = OUT / "html" / "infos", OUT / "pdf" / "infos", OUT / "txt" / "infos"
    for d in (html_dir, pdf_dir, txt_dir):
        d.mkdir(parents=True, exist_ok=True)
    for row in rows:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", unquote(row["page"].split("?q=")[-1]))[:120]
        row["fichier"] = stem
        html_path = html_dir / f"{stem}.html"
        if not html_path.exists():
            html_path.write_bytes(client.get(row["page"]).content)
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        body = soup.select_one(".node .content") or soup.select_one(".node") or soup.select_one("#content")
        text = body.get_text(NL, strip=True) if body else ""
        pdfs = list(dict.fromkeys(urljoin(BASE, a["href"]) for a in (body or soup).find_all("a", href=True)
                                  if a["href"].lower().endswith(".pdf")))
        row["pdf"] = " ".join(pdfs)
        for url in pdfs:
            pdf = pdf_dir / local_name(url)
            if not pdf.exists():
                try:
                    pdf.write_bytes(client.get(url).content)
                except requests.HTTPError as e:
                    logging.warning("Échec %s : %s", url, e)
                    continue
            ptxt = txt_dir / (pdf.stem + ".pdf.txt")
            if not ptxt.exists():
                pdf_to_text(pdf, ptxt)
            if ptxt.exists():
                text += NL + chr(12) + ptxt.read_text(encoding="utf-8", errors="replace")
        (txt_dir / f"{stem}.txt").write_text(text, encoding="utf-8")


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def local_name(url):
    return re.sub(r"[^A-Za-z0-9._-]", "_", unquote(Path(urlparse(url).path).name))


def download(client, rows, subdir):
    pdf_dir, txt_dir = OUT / "pdf" / subdir, OUT / "txt" / subdir
    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if not row.get("url"):
            continue
        name = local_name(row["url"])
        pdf = pdf_dir / name
        if not pdf.exists():
            try:
                content = client.get(row["url"]).content
            except requests.HTTPError as e:
                logging.warning("Échec %s : %s", row["url"], e)
                continue
            pdf.write_bytes(content)
        txt = txt_dir / (pdf.stem + ".txt")
        if not txt.exists():
            pdf_to_text(pdf, txt)


def pdf_to_text(pdf, txt):
    """Texte du PDF avec pdfplumber ; repli sur pdftotext (Git for Windows) s'il est absent."""
    try:
        if pdfplumber is not None:
            with pdfplumber.open(pdf) as doc:
                text = "\n\f".join((page.extract_text(layout=True) or "") for page in doc.pages)
            txt.write_text(text, encoding="utf-8")
        elif shutil.which("pdftotext"):
            subprocess.run(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf), str(txt)], check=True)
        else:
            raise RuntimeError("ni pdfplumber ni pdftotext disponibles")
    except Exception as error:  # PDF corrompu ou scanné : on le signale, sans bloquer
        logging.warning("Extraction texte impossible %s : %s", pdf.name, error)
        return
    if len(txt.read_text(encoding="utf-8").strip()) < 50:
        logging.warning("Texte vide ou quasi vide (PDF scanné ?) : %s", pdf.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=("index", "download", "infos", "all"), default="all")
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--start-page", type=int, default=0, help="reprise du parcours infos à cette page")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=ROOT / "logs" / "collecte_ost_cmf.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    if pdfplumber is None:
        logging.warning("pdfplumber absent : repli sur pdftotext")
        print("Attention : pdfplumber absent (pip install pdfplumber), repli sur pdftotext.")
    client = Client()
    res_path, aug_path = OUT / "index_resolutions.csv", OUT / "index_augmentations.csv"
    try:
        if args.step in ("index", "all"):
            res = crawl_resolutions(client, args.max_pages)
            write_csv(res_path, res, ["societe", "categorie", "annee", "url"])
            aug = crawl_augmentations(client, args.max_pages)
            write_csv(aug_path, aug, ["date_avis", "societe", "page", "url"])
        if args.step in ("download", "all"):
            res = [r for r in read_csv(res_path) if "ADMISES A LA COTE" in r["categorie"].upper()
                   and "NON ADMISES" not in r["categorie"].upper() and r["annee"] in YEARS]
            print(f"PDF résolutions à télécharger : {len(res)}")
            download(client, res, "resolutions")
            aug = read_csv(aug_path)
            print(f"PDF augmentations à télécharger : {len(aug)}")
            download(client, aug, "augmentations")
        if args.step == "infos":
            kept, other = crawl_infos(client, args.max_pages, args.start_page)
            if args.start_page > 0:
                # Reprise : on fusionne avec l'index existant (dédoublonné par page).
                def merge(path, new):
                    old = read_csv(path) if path.exists() else []
                    seen = {r["page"] for r in new}
                    return new + [r for r in old if r["page"] not in seen]
                other = merge(OUT / "index_infos_hors_univers.csv", other)
                kept = merge(OUT / "index_infos.csv", kept)
            write_csv(OUT / "index_infos_hors_univers.csv", other, ["date_publication", "ticker", "titre", "page"])
            write_csv(OUT / "index_infos.csv", kept, ["date_publication", "ticker", "titre", "page", "fichier", "pdf"])
            print(f"Communiqués retenus : {len(kept)} (hors univers : {len(other)})")
            fetch_infos(client, kept)  # documents déjà en cache non retéléchargés
            write_csv(OUT / "index_infos.csv", kept, ["date_publication", "ticker", "titre", "page", "fichier", "pdf"])
    except Stop as e:
        logging.error("%s", e)
        print(e)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
