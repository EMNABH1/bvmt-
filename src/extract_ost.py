"""Extraction des opérations sur titres depuis les textes CMF, puis vérification sur les cours.

Entrées :
  data/ost_cmf/index_resolutions.csv, index_augmentations.csv, txt/<rubrique>/*.txt
      (produits par src/collect_ost_cmf.py)
  mes_exports/<CODE>.csv   cours bruts ilboursa (src/collect_ilboursa.py)
  data/ost_manuelles.csv   (facultatif) opérations relevées à la main dans une autre source
                           publique : ticker,date_paiement,type,montant_ou_parite,document_source

Sortie : data/operations_sur_titres.csv
  ticker, date, type, montant_ou_parite, document_source, date_detachement, verification
  - date : date de mise en paiement (dividende) ou date indiquée dans l'avis (attribution)
  - type : dividende | attribution_gratuite
  - montant_ou_parite : dinars par action (dividende) ou « n:m » = n actions gratuites pour m anciennes
Règle : aucune valeur n'est devinée. Un champ illisible reste vide et le document est listé
dans data/ost_cmf/documents_a_verifier.csv.
"""
import csv
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pandas as pd

from ajustements_ilboursa import cumulative_factor
from univers import fold, ticker_for_name, ticker_in_text

ROOT = Path(__file__).resolve().parents[1]
CMF = ROOT / "data" / "ost_cmf"
EXPORTS = ROOT / "data" / "cours_corriges"  # mes_exports + data/corrections_donnees.csv (src/corriger_donnees.py)
OUT = ROOT / "data" / "operations_sur_titres.csv"
TO_CHECK = CMF / "documents_a_verifier.csv"
MANUAL = ROOT / "data" / "ost_manuelles.csv"
START, END = date(2023, 1, 1), date(2026, 11, 19)

MONTHS = {"janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
          "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12}
NUM_WORDS = {"un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7,
             "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14,
             "quinze": 15, "seize": 16, "vingt": 20}


# ---------------------------------------------------------------------------- texte
DATE_RX = (r"(\d{1,2})\s*(?:er)?\s+(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|"
           r"octobre|novembre|decembre)\s+(\d{4})|(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})")


def parse_date(m):
    try:
        if m.group(1):
            return date(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
        return date(int(m.group(6)), int(m.group(5)), int(m.group(4)))
    except ValueError:
        return None


def to_float(s):
    return float(s.replace(" ", "").replace(",", "."))


AMOUNT = r"(\d{1,3}(?:[.,]\d{1,3})?)\s*(dinars?|dt|tnd|d|millimes?)\b"
DIV_RXS = [
    # « un dividende de 0.625 TND par action », « dividende brut de 6,000 dinars par action »
    re.compile(r"dividendes?\s+(?:brut\s+|net\s+|ordinaire\s+)?(?:global\s+)?(?:de\s+|d'un montant de\s+|s'elevant a\s+)?"
               + AMOUNT + r"[^.;]{0,25}?\bpar\s+(?:action|titre)"),
    # « ... soit 0,350 dinar par action » à proximité de « dividende »
    re.compile(r"dividende[^.]{0,200}?(?:soit|a raison de|de)\s+" + AMOUNT + r"[^.;]{0,25}?\bpar\s+(?:action|titre)"),
    # « le dividende par action est fixé à 6,000 dinars »
    re.compile(r"dividende\s+(?:brut\s+|net\s+)?par\s+(?:action|titre)[^.]{0,40}?(?:fixe|arrete|porte|egal)e?s?\s+a\s+" + AMOUNT),
    # « fixe le montant des dividendes ... à 3,600 Dinars par action » / « le dividende à 8,9DT pour chacune des actions »
    re.compile(r"dividendes?[^.]{0,120}?\ba\s+" + AMOUNT + r"[^.;]{0,25}?\b(?:par\s+(?:action|titre)|pour\s+chacune)"),
    # « montant de dividende par action de 0,330 dinars » / « dividende par action : 0,500 dinar »
    re.compile(r"dividende\s+(?:brut\s+|net\s+)?par\s+(?:action|titre)\s*(?:de|:|=)\s*" + AMOUNT),
    # « quatre dinars deux cents millimes par action (4,2 Dinars) »
    re.compile(r"dividendes?[^.]{0,160}?par\s+(?:action|titre)\s*\(\s*" + AMOUNT + r"\s*\)"),
    # « cinq dinars (5,000 DT) par action »
    re.compile(r"dividendes?[^.]{0,160}?\(\s*" + AMOUNT + r"\s*\)\s*par\s+(?:action|titre)"),
]
# « DIVIDENDE TOTAL PAR ACTION = 0 D,880 » (notation dinars D millimes)
DIV_DMILL_RX = re.compile(r"dividende[^.=]{0,30}par\s+action\s*=\s*(\d{1,3})\s*d\s*,\s*(\d{3})")
# « Le détachement du dividende se fera le 12 mai 2026 » (peut précéder la mise en paiement)
DETACH_RX = re.compile(r"detach\w*[^.]{0,80}?(?:" + DATE_RX + ")")
PAY_RX = re.compile(r"(?:mise?s?\s+en\s+paiement|paiement\s+(?:de\s+ces|des|du)\s+dividendes?|payables?|paye"
                    r"|date\s+d[ue']\s*(?:la\s+)?paiement|(?:le\s+)?paiement\s+(?:sera|se\s+fera|aura\s+lieu|interviendra))"
                    r"[^.]{0,120}?(?:" + DATE_RX + ")")
NUM = r"(?:(\d+)|(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|vingt))(?:\s*\(\s*\d+\s*\))?"
BONUS_RX = re.compile(r"(?:raison\s+d[e']\s*)?" + NUM + r"\s+actions?\s+(?:nouvelles?\s+)?(?:gratuites?\s+)?(?:nouvelles?\s+)?"
                      r"pour\s+" + NUM + r"\s+(?:actions?\s+)?anciennes?")
BONUS_DATE_RX = re.compile(r"(?:jouissance|a\s+partir\s+du|a\s+compter\s+du|date\s+d'attribution)[^.]{0,60}?(?:" + DATE_RX + ")")


def num(m, i):
    """Nombre en chiffres (groupe i) ou en lettres (groupe i+1), avec « une (1) » -> 1."""
    raw = m.group(i) or m.group(i + 1)
    whole = m.group(0)
    paren = re.search(r"\(\s*(\d+)\s*\)", whole)
    if raw is None:
        return None
    return int(raw) if raw.isdigit() else NUM_WORDS.get(raw)


def extract_document(text):
    """Retourne (événements, problèmes). Un événement = dict(type, date, valeur)."""
    t = fold(text)
    events, problems = [], []
    mentions_div = re.search(r"(distribu\w*|attribu\w*)[^.]{0,80}dividende|dividende[^.]{0,80}(distribu|par action)", t)
    amounts = []
    for rx in DIV_RXS:
        for m in rx.finditer(t):
            v = to_float(m.group(1))
            if m.group(2).startswith("millime"):
                v /= 1000
            if 0 < v < 100:
                amounts.append(round(v, 4))
    for m in DIV_DMILL_RX.finditer(t):
        amounts.append(round(int(m.group(1)) + int(m.group(2)) / 1000, 4))
    amounts = sorted(set(amounts))
    pay_dates = sorted({d for m in PAY_RX.finditer(t) if (d := parse_date_from(m))})
    detach_dates = sorted({d for m in DETACH_RX.finditer(t) if (d := first_date(m))})
    if mentions_div or amounts:
        if re.search(r"(aucun|pas de)\s+(distribution\s+de\s+)?dividende|ne\s+pas\s+distribuer", t) and not amounts:
            pass  # l'AG décide de ne pas distribuer : rien à enregistrer
        else:
            amount = amounts[0] if len(amounts) == 1 else None
            pay = pay_dates[0] if len(pay_dates) == 1 else None
            if amounts or pay_dates:
                events.append({"type": "dividende", "date": pay, "valeur": amount,
                               "detach": detach_dates[0] if len(detach_dates) == 1 else None})
            if len(amounts) != 1:
                problems.append(f"montant de dividende {'ambigu ' + str(amounts) if amounts else 'illisible'}")
            if len(pay_dates) != 1:
                problems.append(f"date de mise en paiement {'ambiguë ' + str([str(d) for d in pay_dates]) if pay_dates else 'illisible'}")
    if "gratuit" in t:
        parities = sorted({(num(m, 1), num(m, 3)) for m in BONUS_RX.finditer(t)} - {(None, None)})
        parities = [p for p in parities if p[0] and p[1]]
        bdates = sorted({d for m in BONUS_DATE_RX.finditer(t) if (d := parse_date_from(m))})
        if parities or re.search(r"attribution\s+(d'actions\s+)?gratuite|actions\s+gratuites", t):
            parity = f"{parities[0][0]}:{parities[0][1]}" if len(parities) == 1 else None
            bdate = bdates[0] if len(bdates) == 1 else None
            events.append({"type": "attribution_gratuite", "date": bdate, "valeur": parity})
            if len(parities) != 1:
                problems.append(f"parité d'attribution {'ambiguë ' + str(parities) if parities else 'illisible'}")
            if len(bdates) != 1:
                problems.append("date d'attribution " + ("ambiguë" if bdates else "absente du document"))
    return events, problems


def first_date(m):
    """Première date du texte capturé (pour « détachement ... le 12 mai 2026 et ... »)."""
    dm = re.search(DATE_RX, m.group(0))
    d = parse_date(dm) if dm else None
    return d if d and START <= d <= date(2027, 6, 30) else None


def parse_date_from(m):
    """Les groupes de DATE_RX sont en fin de motif : on relit la date dans le texte capturé."""
    dm = re.search(DATE_RX, m.group(0)[-40:]) or re.search(DATE_RX, m.group(0))
    d = parse_date(dm) if dm else None
    return d if d and START <= d <= date(2027, 6, 30) else None


def dedupe_cmf_dividends(ops):
    """Un même dividende CMF peut figurer dans la résolution d'AG et dans le communiqué de paiement.
    Deux lignes CMF du même titre, à moins de 30 jours, de même montant (ou l'un vide) = un seul
    dividende. On garde : date de détachement explicite du document > date vérifiée sur les cours
    > montant renseigné."""
    ops = ops.copy()
    is_cmf = ops["type"].eq("dividende") & ops["document_source"].str.contains("cmf.tn")
    ops["_d"] = pd.to_datetime(ops["date_detachement_document"].where(ops["date_detachement_document"].ne(""),
                                ops["date_detachement"].where(ops["date_detachement"].ne(""), ops["date"])),
                               errors="coerce")
    ops["_prio"] = (ops["date_detachement_document"].ne("").astype(int) * 4
                    + ops["verification"].str.startswith("confirmé").astype(int) * 2
                    + ops["montant_ou_parite"].ne("").astype(int))
    drop = set()
    for _, g in ops[is_cmf & ops["_d"].notna()].groupby("ticker"):
        g = g.sort_values("_prio", ascending=False)
        kept = []
        for i, r in g.iterrows():
            dup = any(abs((r["_d"] - k["_d"]).days) <= 30 and (not r["montant_ou_parite"] or not k["montant_ou_parite"]
                      or abs(float(r["montant_ou_parite"]) - float(k["montant_ou_parite"])) < 1e-6) for k in kept)
            if dup:
                drop.add(i)
            else:
                kept.append(r)
    if drop:
        print(f"{len(drop)} doublon(s) CMF retiré(s) : " + ", ".join(
            f"{ops.at[i, 'ticker']} {ops.at[i, '_d'].date()}" for i in sorted(drop)))
    return ops.drop(index=list(drop)).drop(columns=["_d", "_prio"])


# ---------------------------------------------------------------------------- cours
def load_prices(ticker):
    path = EXPORTS / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str)
    if df.empty:
        return None
    for col in ("ouverture", "cloture", "volume"):
        df[col] = pd.to_numeric(df[col].str.replace(",", ".").str.replace(" ", ""), errors="coerce")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return df.set_index("date").sort_index()


def expected_ratio(ev, prev_close, f=1.0):
    """f : facteur déjà appliqué par ilboursa à ce jour (dividende par ancienne action -> D / f)."""
    if ev["type"] == "dividende":
        return (prev_close - float(ev["montant_ou_parite"]) / f) / prev_close
    n, m = map(int, ev["montant_ou_parite"].split(":"))
    return m / (m + n)


def verify(ev, prices):
    """Cherche la séance de détachement autour de la date de paiement / d'attribution.

    Fenêtre : 10 séances avant à 5 séances après. Pour chaque séance t on compare le rapport
    observé cloture(t)/cloture(t-1) (et ouverture(t)/cloture(t-1)) au rapport attendu.
    Retenue : la séance qui minimise l'écart ; « confirmé » si l'écart est inférieur à la moitié
    de la baisse attendue (et à 1,5 point de %).
    """
    if prices is None or prices["cloture"].dropna().empty:
        return "", "cours indisponibles"
    if not ev["montant_ou_parite"] or not ev["date"]:
        return "", "non vérifiable (champ manquant)"
    close = prices["cloture"].dropna()
    # Date de détachement donnée par le document si elle existe, sinon date de paiement.
    ref = pd.Timestamp(ev.get("date_detachement_document") or ev["date"])
    pos = close.index.searchsorted(ref)
    lo, hi = max(1, pos - 10), min(len(close) - 1, pos + 5)
    if lo > hi:
        return "", "date hors de la période des cours"
    best = None
    for i in range(lo, hi + 1):
        prev = close.iloc[i - 1]
        exp = expected_ratio(ev, prev, cumulative_factor(ev["ticker"], close.index[i - 1]))
        obs_c = close.iloc[i] / prev
        op = prices["ouverture"].get(close.index[i])
        obs_o = op / prev if op and not np.isnan(op) else obs_c  # indicatif seulement
        err = abs(obs_c - exp)  # clôtures uniquement : ouvertures ilboursa peu fiables
        if best is None or err < best[0]:
            best = (err, close.index[i], exp, obs_c, obs_o)
    err, day, exp, obs_c, obs_o = best
    drop = 1 - exp
    detail = f"attendu {exp - 1:+.2%}, observé clôture {obs_c - 1:+.2%} (ouverture indicative {obs_o - 1:+.2%})"
    if err <= min(0.5 * drop, 0.015) or (drop < 0.01 and err <= 0.005):
        lag = (day - ref).days
        return day.date().isoformat(), f"confirmé ({detail}; écart à la date du document {lag:+d} j)"
    return "", f"non confirmé : meilleure séance {day.date()} ({detail})"


# ---------------------------------------------------------------------------- principal
def local_stem(url):
    return re.sub(r"[^A-Za-z0-9._-]", "_", unquote(Path(urlparse(url).path).name)).rsplit(".", 1)[0]


def documents():
    res = pd.read_csv(CMF / "index_resolutions.csv", dtype=str)
    res = res[res["categorie"].str.upper().eq("SOCIETES ADMISES A LA COTE")
              & res["annee"].isin([str(y) for y in range(START.year, END.year + 1)])]
    aug = pd.read_csv(CMF / "index_augmentations.csv", dtype=str)
    for _, r in res.iterrows():
        yield r["societe"], "resolutions", r["url"]
    for _, r in aug.iterrows():
        yield r["societe"], "augmentations", r["url"]
    infos_path = CMF / "index_infos.csv"
    if infos_path.exists():
        infos = pd.read_csv(infos_path, dtype=str).fillna("")
        for _, r in infos.iterrows():
            if r.get("fichier"):
                yield ("@" + r["ticker"]), "infos", r["page"] + "#" + r["fichier"]


def main():
    rows, to_check, unmapped = [], [], set()
    for societe, rubrique, url in documents():
        if not isinstance(url, str) or not url:
            continue
        ticker = societe[1:] if societe.startswith("@") else (ticker_for_name(societe) or ticker_in_text(societe))
        if ticker is None:
            unmapped.add(societe)
            continue
        if rubrique == "infos":
            url, stem = url.rsplit("#", 1)
            txt = CMF / "txt" / "infos" / f"{stem}.txt"
        else:
            txt = CMF / "txt" / rubrique / f"{local_stem(url)}.txt"
        if not txt.exists():
            to_check.append({"ticker": ticker, "document_source": url, "probleme": "texte absent (non téléchargé)"})
            continue
        text = txt.read_text(encoding="utf-8", errors="replace")
        if len(text.strip()) < 50:
            to_check.append({"ticker": ticker, "document_source": url, "probleme": "PDF sans texte (scanné ?)"})
            continue
        events, problems = extract_document(text)
        for ev in events:
            if ev["date"] and not (START <= ev["date"] <= END):
                continue
            rows.append({"ticker": ticker, "date": ev["date"].isoformat() if ev["date"] else "",
                         "type": ev["type"], "montant_ou_parite": "" if ev["valeur"] is None else str(ev["valeur"]),
                         "document_source": url,
                         "date_detachement_document": ev["detach"].isoformat() if ev.get("detach") else ""})
        for p in problems:
            to_check.append({"ticker": ticker, "document_source": url, "probleme": p})

    if MANUAL.exists():
        manual = pd.read_csv(MANUAL, dtype=str).fillna("")
        for _, r in manual.iterrows():
            rows.append({"ticker": r["ticker"], "date": r["date_paiement"], "type": r["type"],
                         "montant_ou_parite": r["montant_ou_parite"], "document_source": r["document_source"]})

    ops = pd.DataFrame(rows, columns=["ticker", "date", "type", "montant_ou_parite", "document_source",
                                      "date_detachement_document"]).fillna("")
    ops = ops.drop_duplicates(subset=["ticker", "date", "type", "montant_ou_parite"])
    cache = {}
    det, ver = [], []
    for _, ev in ops.iterrows():
        if ev["ticker"] not in cache:
            cache[ev["ticker"]] = load_prices(ev["ticker"])
        d, v = verify(ev, cache[ev["ticker"]])
        det.append(d)
        ver.append(v)
    # Date de détachement explicite dans le document CMF : c'est elle qui est retenue,
    # même si les cours ne montrent pas la baisse (l'écart est signalé dans « verification »).
    for k, (doc_d, d, v) in enumerate(zip(ops["date_detachement_document"], det, ver)):
        if doc_d:
            det[k] = doc_d
            if d != doc_d:
                why = "cours non concordants" if v.startswith("non confirmé") else "non vérifiée sur les cours"
                ver[k] = f"date du document retenue ; {why} : {v}"
    ops["date_detachement"], ops["verification"] = det, ver
    # Dividendes 2023-2025 absents du CMF : montant ilboursa + date détectée sur les cours
    # (src/dividendes_ilboursa.py). Lignes « détecté » (date exacte) et « approximatif » (plusieurs séances compatibles,
    # séance d'écart minimal) ajoutées.
    ilb_path = ROOT / "data" / "dividendes_ilboursa.csv"
    if ilb_path.exists():
        ilb = pd.read_csv(ilb_path, dtype=str).fillna("")
        cmf_years = {(t, (d or dd)[:4]) for t, d, dd, ty in zip(ops["ticker"], ops["date"], ops["date_detachement_document"],
                                                             ops["type"]) if ty == "dividende" and (d or dd)}
        extra = [{
            "ticker": r["ticker"], "date": r["date_detachement"], "type": "dividende",
            "montant_ou_parite": r["montant"],
            "document_source": f"{r['source']} : https://www.ilboursa.com/marches/societe/{r['ticker']} (exercice {r['exercice']})",
            "date_detachement": r["date_detachement"],
            "verification": f"{'détecté' if r['statut'] == 'détecté' else 'approximatif'} (baisse {float(r['baisse_observee']):.2%} pour {float(r['baisse_attendue']):.2%} attendus, "
                            f"écart {r['ecart_points']} pt, {r['nb_seances_compatibles']} séance(s) compatible(s))",
        } for _, r in ilb.iterrows() if (r["statut"] == "détecté" or r["statut"].startswith("approximatif")) and (r["ticker"], r["annee_paiement"]) not in cmf_years]
        if extra:
            ops = pd.concat([ops, pd.DataFrame(extra)], ignore_index=True)
    ops = dedupe_cmf_dividends(ops)
    ops = ops.sort_values(["ticker", "date", "type"])
    ops.to_csv(OUT, index=False, encoding="utf-8")
    pd.DataFrame(to_check, columns=["ticker", "document_source", "probleme"]).to_csv(TO_CHECK, index=False, encoding="utf-8")
    print(f"{len(ops)} opérations -> {OUT}")
    print(ops["verification"].str.split(" ").str[0].value_counts().to_string())
    print(f"{len(to_check)} problèmes -> {TO_CHECK}")
    if unmapped:
        print("Sociétés CMF hors univers (ignorées) :", ", ".join(sorted(unmapped)))


if __name__ == "__main__":
    main()
