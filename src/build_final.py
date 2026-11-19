"""Jeu de données final (section 4 du cahier des charges) : ajustement, filtres, rapport qualité.

Entrées :
  mes_exports/<CODE>.csv            cours bruts ilboursa (collect_ilboursa.py), PX1 = Tunindex
  data/operations_sur_titres.csv    dividendes et attributions documentés (extract_ost.py)
  data/sauts_suspects.csv           sauts détectés et confirmés (detect_sauts.py)
  data/dividendes_ilboursa.csv      dividendes ilboursa non ajustés (statuts, pour le rapport)
  data/sauts_a_verifier.csv         sauts très suspects sans confirmation (verifier_sauts.py)
  data/ost_cmf/documents_a_verifier.csv

Ajustement (en remontant le temps, clôtures uniquement) :
  - dividende D détaché à la séance t : facteur (P(t-1) - D) / P(t-1), P(t-1) = clôture brute
    de la séance cotée précédente ;
  - attribution n pour m : facteur m / (m + n) ; division k pour 1 : facteur 1 / k ;
  - chaque cours antérieur à t est multiplié par le produit des facteurs des opérations
    postérieures. Le dernier cours reste le cours réel.
Opérations utilisées : dividendes avec montant et date de détachement (CMF, ou
« ilboursa + détection » non ambigu) ; attributions/divisions confirmées (CMF ou investing)
avec une parité documentée. Tout le reste est listé dans le rapport, non ajusté.

Filtres : historique ≥ 2 ans (première à dernière cotation) ; liquidité ≥ 60 % des séances
du Tunindex (de la première cotation à la fin) avec un volume > 0.

Sorties (data/final/) : prix_ajustes.csv (date, ticker, close_brut, close_ajuste, volume,
  seance_neutralisee), tunindex.csv, titres_retenus.csv, rapport_qualite.md
Aucun rendement ni indicateur n'est calculé.
"""
import csv
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ajustements_ilboursa import cumulative_factor

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "cours_corriges"  # mes_exports + data/corrections_donnees.csv (src/corriger_donnees.py)
DATA = ROOT / "data"
FINAL = DATA / "final"
START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-11-19")
MIN_YEARS = 2
MIN_TRADING_RATIO = 0.60
BIG_MOVE = 0.25
INVESTING = {"BIAT": ROOT.parent / "Biat.csv.csv", "SFBT": ROOT.parent / "Sfbt.csv.csv"}


# ----------------------------------------------------------------------------- lecture
def names():
    with (DATA / "codes_ilboursa.csv").open(encoding="utf-8-sig", newline="") as f:
        return {r["code"].strip().upper(): r["nom"] for r in csv.DictReader(f) if r.get("code", "").strip()}


def load(code):
    df = pd.read_csv(EXPORTS / f"{code}.csv", sep=";", encoding="utf-8-sig", dtype=str)
    if df.empty:
        return pd.DataFrame(columns=["cloture", "volume"], dtype=float)
    out = pd.DataFrame({c: pd.to_numeric(df[c].str.replace(",", ".").str.replace(" ", ""), errors="coerce")
                        for c in ("cloture", "volume")})
    out.index = pd.to_datetime(df["date"], format="%d/%m/%Y")
    out = out[(out.index >= START) & (out.index <= END)]
    return out.dropna(subset=["cloture"]).sort_index()


def read(path, **kw):
    return pd.read_csv(path, dtype=str, **kw).fillna("") if path.exists() else pd.DataFrame()


def md_table(df):
    if df is None or len(df) == 0:
        return "_Aucun._"
    return df.to_markdown(index=False)


# ----------------------------------------------------------------------------- opérations
def operations_to_apply(prices):
    """Liste des opérations appliquées (dict) et des opérations écartées (dict, motif)."""
    applied, skipped = [], []
    ops = read(DATA / "operations_sur_titres.csv")
    for _, o in ops.iterrows():
        if "cmf.tn" in o["document_source"]:
            src = "CMF"
        elif o["document_source"].startswith("ilboursa") and o["verification"].startswith("approximatif"):
            n = re.search(r"(\d+)(?:\.0)? séance", o["verification"])
            src = f"ilboursa + détection (date approximative, {n.group(1) if n else '?'} séances compatibles)"
        elif o["document_source"].startswith("ilboursa"):
            src = "ilboursa (montant) + détection (date)"
        else:
            src = o["document_source"]
        base = {"ticker": o["ticker"], "type": o["type"], "valeur": o["montant_ou_parite"], "source": src,
                "document": o["document_source"], "verification": o["verification"]}
        if o["type"] == "dividende":
            if not o["montant_ou_parite"] or not o["date_detachement"]:
                skipped.append({**base, "date": o["date_detachement"] or o["date"],
                                "motif": "montant ou date de détachement manquant"})
                continue
            applied.append({**base, "date": o["date_detachement"]})
        elif o["type"] == "attribution_gratuite":
            # Les attributions passent par les sauts confirmés (parité retenue) ; une attribution
            # CMF n'est appliquée ici que si sa date de détachement a été vérifiée sur les cours.
            if o["montant_ou_parite"] and o["date_detachement"] and o["verification"].startswith("confirmé"):
                applied.append({**base, "date": o["date_detachement"]})
            else:
                skipped.append({**base, "date": o["date_detachement"] or o["date"],
                                "motif": "attribution sans parité ou sans date vérifiée"})
    # Opérations lues et validées dans les documents (data/operations_validees.csv)
    for _, v in read(DATA / "operations_validees.csv").iterrows():
        applied.append({"ticker": v["ticker"], "type": v["type"], "valeur": v["valeur"], "source": v["source"],
                        "document": v["document"], "verification": v["justification"], "date": v["date"]})
    sauts = read(DATA / "sauts_suspects.csv")
    if len(sauts):
        for _, s in sauts[sauts["confirme"].ne("non")].iterrows():
            if not s["parite_retenue"]:
                continue  # confirmé par un document de dividende : traité via les dividendes
            m = re.match(r"(\d+)/(\d+) \((.*)\)", s["parite_retenue"])
            num, den, label = int(m.group(1)), int(m.group(2)), m.group(3)
            if label.startswith("division"):
                value, typ = f"{num}:1", "division"
            else:
                value, typ = f"{num - den}:{den}", "attribution_gratuite"
            dup = any(a["ticker"] == s["ticker"] and a["type"] == typ and abs(
                (pd.Timestamp(a["date"]) - pd.Timestamp(s["date"])).days) <= 14 for a in applied)
            if not dup:
                applied.append({"ticker": s["ticker"], "type": typ, "valeur": value, "date": s["date"],
                                "source": f"saut confirmé ({s['confirme']}), parité : {s['source_parite']}",
                                "document": s["document_confirmation"] or s["controle_investing"],
                                "verification": s["controle_investing"] or s["document_confirmation"]})
    return applied, skipped


def factor(op, close):
    day = pd.Timestamp(op["date"])
    before = close[close.index < day]
    if before.empty:
        return None, "aucune cotation avant la date"
    if op["type"] == "dividende":
        p = before.iloc[-1]
        f = cumulative_factor(op["ticker"], before.index[-1])
        d = float(op["valeur"]) / f  # dividende par ancienne action ramené aux unités ilboursa
        if not 0 < d < p:
            return None, f"dividende {d} incohérent avec P(t-1) = {p}"
        dl = f"{float(op['valeur']):g}/{f:g}" if f != 1.0 else f"{d:g}"
        return (p - d) / p, f"(P(t-1) − D)/P(t-1) = ({p:g} − {dl})/{p:g}"
    if op["type"] == "attribution_gratuite":
        n, m = map(int, op["valeur"].split(":"))
        return m / (m + n), f"m/(m+n) = {m}/{m + n}"
    if op["type"] == "augmentation_mixte":
        # « gratuite n:m ; numeraire k:m a X » : pour m anciennes, n gratuites et k souscrites à X.
        # Facteur théorique (TERP) : (m·P + k·X) / ((m + n + k)·P), P = P(t-1) en unités ilboursa.
        g = re.search(r"gratuite\s+(\d+):(\d+)", op["valeur"])
        k = re.search(r"numeraire\s+(\d+):(\d+)\s+a\s+([\d.]+)", op["valeur"])
        p = before.iloc[-1]
        n, m = (int(g.group(1)), int(g.group(2))) if g else (0, int(k.group(2)))
        kk, x = (int(k.group(1)), float(k.group(3))) if k else (0, 0.0)
        terp = (m * p + kk * x) / (m + n + kk)
        return terp / p, f"TERP/P(t-1) = ({m}×{p:g} + {kk}×{x:g})/({m + n + kk}×{p:g}) = {terp:.4f}/{p:g}"
    if op["type"] == "division":
        k = int(op["valeur"].split(":")[0])
        return 1 / k, f"1/{k}"
    return None, f"type inconnu {op['type']}"


def adjust(close, ops):
    adj_factor = pd.Series(1.0, index=close.index)
    log = []
    for op in sorted(ops, key=lambda o: o["date"]):
        f, how = factor(op, close)
        if f is None:
            log.append({**op, "facteur": "", "calcul": how, "applique": "non"})
            continue
        adj_factor[adj_factor.index < pd.Timestamp(op["date"])] *= f
        log.append({**op, "facteur": round(f, 6), "calcul": how, "applique": "oui"})
    return close * adj_factor, log


# ----------------------------------------------------------------------------- principal
def main():
    FINAL.mkdir(parents=True, exist_ok=True)
    noms = names()
    index = load("PX1")
    calendar = index.index
    pd.DataFrame({"date": calendar.date, "close": index["cloture"].values}).to_csv(
        FINAL / "tunindex.csv", index=False, encoding="utf-8")

    applied, skipped = operations_to_apply(None)
    by_ticker = {}
    for op in applied:
        by_ticker.setdefault(op["ticker"], []).append(op)

    sc_path = ROOT / "data" / "titres_sans_cotation.csv"
    sans_cotation = {}
    if sc_path.exists():
        sc = pd.read_csv(sc_path, dtype=str)
        sans_cotation = {r["ticker"]: f"{r['motif']} (preuve : {r['preuve']})" for _, r in sc.iterrows()}
    rows, status, adj_log, big_moves, missing = [], [], [], [], []
    prices_raw = {}
    codes = sorted(set(noms) | {p.stem for p in EXPORTS.glob("*.csv")} - {"PX1"})
    for code in codes:
        name = noms.get(code, "")
        if not (EXPORTS / f"{code}.csv").exists():
            status.append({"ticker": code, "nom": name, "retenu": "non", "motif": "cours non collectés"})
            continue
        df = load(code)
        if df.empty:
            # Titre absent des téléchargements : motif documenté (dernière cotation vérifiée sur ilboursa)
            # dans data/titres_sans_cotation.csv, sinon « aucune donnée ilboursa ».
            motif = sans_cotation.get(code, "aucune donnée de cours sur ilboursa")
            status.append({"ticker": code, "nom": name, "retenu": "non", "motif": motif})
            continue
        close, vol = df["cloture"], df["volume"].fillna(0)
        prices_raw[code] = close
        adj, log = adjust(close, by_ticker.get(code, []))
        adj_log += [{"ticker": code, **{k: v for k, v in l.items() if k != "ticker"}} for l in log]

        first, last = close.index[0], close.index[-1]
        sessions = calendar[(calendar >= first) & (calendar <= END)]
        quoted = close.index.intersection(sessions)
        n_missing = len(sessions) - len(quoted)
        traded = int((vol.reindex(sessions).fillna(0) > 0).sum())
        ratio = traded / len(sessions) if len(sessions) else 0.0
        years = (last - first).days / 365.25
        motif = []
        if years < MIN_YEARS:
            motif.append(f"historique {years:.2f} ans < {MIN_YEARS} ans")
        if ratio < MIN_TRADING_RATIO:
            motif.append(f"liquidité {ratio:.0%} < {MIN_TRADING_RATIO:.0%} de séances avec volume > 0")
        kept = not motif
        missing.append({"ticker": code, "premiere_cotation": first.date(), "derniere_cotation": last.date(),
                        "seances_tunindex": len(sessions), "seances_cotees": len(quoted),
                        "jours_manquants": n_missing, "seances_avec_volume": traded,
                        "ratio_jours_echanges": round(ratio, 3)})
        status.append({"ticker": code, "nom": name, "retenu": "oui" if kept else "non", "motif": "; ".join(motif),
                       "premiere_cotation": first.date(), "derniere_cotation": last.date(),
                       "historique_ans": round(years, 2), "seances_cotees": len(quoted),
                       "jours_manquants": n_missing, "ratio_jours_echanges": round(ratio, 3),
                       "nb_ajustements": sum(1 for l in log if l["applique"] == "oui")})

        # Variations journalières > 25 % (sur cours ajustés) non expliquées par une opération.
        r = adj.pct_change().dropna()
        op_days = {pd.Timestamp(o["date"]) for o in by_ticker.get(code, [])}
        for d, v in r[r.abs() > BIG_MOVE].items():
            prev_d = adj.index[adj.index.get_loc(d) - 1]
            i0, i1 = index["cloture"].asof(prev_d), index["cloture"].asof(d)
            big_moves.append({"ticker": code, "date": d.date(), "seance_precedente": prev_d.date(),
                              "cours_brut_prec": close[prev_d], "cours_brut": close[d],
                              "variation_ajustee": round(v, 4), "variation_tunindex": round(i1 / i0 - 1, 4),
                              "retenu": "oui" if kept else "non",
                              "explication": "aucune opération documentée" if d not in op_days else "opération à cette date"})
        if kept:
            for d in close.index:
                rows.append({"date": d.date(), "ticker": code, "close_brut": close[d],
                             "close_ajuste": round(adj[d], 6), "volume": int(vol[d])})

    prices = pd.DataFrame(rows, columns=["date", "ticker", "close_brut", "close_ajuste", "volume"])
    # Séances à baisse suspecte non confirmée (data/seances_neutralisees.csv) : le cours est gardé tel quel,
    # la colonne seance_neutralisee = 1 permet d'exclure le rendement de ce jour dans une variante.
    neut = read(DATA / "seances_neutralisees.csv")
    keys = set(zip(neut["ticker"], neut["date"])) if len(neut) else set()
    prices["seance_neutralisee"] = [int((t, str(d)) in keys) for t, d in zip(prices["ticker"], prices["date"])]
    prices.to_csv(FINAL / "prix_ajustes.csv", index=False, encoding="utf-8")
    st = pd.DataFrame(status)
    st[st["retenu"].eq("oui")].drop(columns=["retenu", "motif"]).to_csv(
        FINAL / "titres_retenus.csv", index=False, encoding="utf-8")
    write_report(st, pd.DataFrame(missing), pd.DataFrame(adj_log), pd.DataFrame(skipped),
                 pd.DataFrame(big_moves), index, prices, prices_raw)
    print(f"Titres retenus : {int(st['retenu'].eq('oui').sum())}/{len(st)} ; lignes de prix : {len(prices)}")
    print(f"Ajustements appliqués : {sum(1 for l in adj_log if l['applique'] == 'oui')} ; variations > 25 % : {len(big_moves)}")


# ----------------------------------------------------------------------------- contrôle investing
def investing_control(prices):
    """Ajusté des seules attributions vs investing.com (qui n'ajuste que les attributions)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from collect_ilboursa import read_investing_prices
    lines = []
    sauts = read(DATA / "sauts_suspects.csv")
    for code, path in INVESTING.items():
        if not path.exists() or not (EXPORTS / f"{code}.csv").exists():
            continue
        close = load(code)["cloture"]
        inv = pd.Series(read_investing_prices(path))
        inv.index = pd.to_datetime(inv.index)
        bonus_only = close.copy()
        conf = sauts[(sauts["ticker"] == code) & sauts["confirme"].ne("non") & sauts["parite_retenue"].ne("")] if len(sauts) else []
        for _, s in (conf.iterrows() if len(conf) else []):
            num, den = map(int, s["parite_retenue"].split(" ")[0].split("/"))
            bonus_only[bonus_only.index < pd.Timestamp(s["date"])] *= den / num
        j = pd.concat([close, bonus_only, inv], axis=1, keys=["brut", "attrib", "inv"], sort=True).dropna()
        raw_ok = ((j["brut"] - j["inv"]).abs() <= 0.011).mean()
        adj_ok = ((j["attrib"] - j["inv"]).abs() <= 0.011).mean()
        lines.append(f"| {code} | {len(j)} | {raw_ok:.1%} | {adj_ok:.1%} | "
                     + ", ".join(f"{s['date']} {s['parite_retenue']}" for _, s in (conf.iterrows() if len(conf) else [])) + " |")
    return ["| Titre | Séances communes | Brut = investing (±0,01) | Ajusté des attributions = investing | Attributions |",
            "|---|---|---|---|---|"] + lines


# ----------------------------------------------------------------------------- rapport
def dividend_bias(divs, st, prices_raw):
    """Dividendes ilboursa approximatifs, non ajustés, et biais estimé par titre."""
    empty = pd.DataFrame()
    if not len(divs):
        return empty, empty, empty, "n/d"
    retenu = dict(zip(st["ticker"], st["retenu"]))
    divs = divs.copy()
    divs["retenu"] = divs["ticker"].map(retenu).fillna("non")
    approx = divs[divs["statut"].str.startswith("approximatif")]
    no_match = divs[divs["montant"].ne("") & (divs["statut"].str.startswith("non détecté")
                                               | divs["statut"].str.startswith("aucune séance"))].copy()
    pv, ratios = [], []
    for _, r in no_match.iterrows():
        p = pd.to_numeric(r.get("cours_veille", ""), errors="coerce")
        if pd.isna(p):
            c = prices_raw.get(r["ticker"])
            if c is not None:
                y = int(r["annee_paiement"])
                w = c[(c.index >= pd.Timestamp(y, 4, 1)) & (c.index <= pd.Timestamp(y, 8, 31))]
                p = w.median() if len(w) else pd.NA
        pv.append(None if pd.isna(p) else round(float(p), 3))
        fac = pd.to_numeric(r.get("facteur_ilboursa", ""), errors="coerce")
        fac = 1.0 if pd.isna(fac) else float(fac)  # dividende par ancienne action -> unités ilboursa
        ratios.append(None if pd.isna(p) else round(float(r["montant"]) / fac / float(p), 4))
    no_match["cours_veille_estime"], no_match["D_sur_P"] = pv, ratios
    # D/P > 10 % : montant ilboursa ou cours douteux (échelle, nominal) ; signalé, à vérifier.
    no_match["a_verifier"] = ["oui (D/P > 10 %)" if r is not None and r > 0.10 else "" for r in ratios]
    rows = []
    for t, g in no_match.groupby("ticker"):
        c = prices_raw.get(t)
        years = (c.index[-1] - c.index[0]).days / 365.25 if c is not None and len(c) > 1 else None
        s = pd.to_numeric(g["D_sur_P"], errors="coerce").sum()
        rows.append({"ticker": t, "retenu": retenu.get(t, "non"), "dividendes_non_ajustes": len(g),
                     "exercices": ", ".join(g["exercice"]), "somme_D_sur_P": round(s, 4),
                     "annees_cotation": round(years, 2) if years else None,
                     "biais_rendement_annualise": f"{s / years:.2%}" if years else "n/d",
                     "a_verifier": "oui (un D/P > 10 %)" if g["a_verifier"].ne("").any() else ""})
    bias = pd.DataFrame(rows)
    if len(bias):
        vals = [s / a for s, a in zip(bias["somme_D_sur_P"], bias["annees_cotation"]) if a]
        bias_median = f"{pd.Series(vals).median():.2%} (sur {len(vals)} titres)" if vals else "n/d"
    else:
        bias_median = "n/d"
    return approx, no_match, bias, bias_median


def write_report(st, missing, adj_log, skipped, big, index, prices, prices_raw):
    kept = st[st["retenu"].eq("oui")]
    excl = st[st["retenu"].eq("non")]
    divs = read(DATA / "dividendes_ilboursa.csv")
    to_check = read(DATA / "sauts_a_verifier.csv")
    docs = read(DATA / "ost_cmf" / "documents_a_verifier.csv")
    sauts = read(DATA / "sauts_suspects.csv")
    applied = adj_log[adj_log["applique"].eq("oui")] if len(adj_log) else adj_log
    approx, no_match, bias, bias_median = dividend_bias(divs, st, prices_raw)
    divs_applied = applied[applied["type"].eq("dividende")] if len(applied) else applied
    n_div_cmf = int(divs_applied["source"].eq("CMF").sum()) if len(applied) else 0
    n_div_approx = int(divs_applied["source"].str.contains("approximative").sum()) if len(applied) else 0
    n_div_exact = len(divs_applied) - n_div_approx
    n_div_approx0 = int(divs_applied["source"].str.contains("approximative, 0 séances").sum()) if len(applied) else 0
    cmf_mismatch = applied[applied["verification"].str.startswith("date du document retenue")] if len(applied) else applied
    L = []
    L += ["# Rapport qualité des données BVMT", "",
          "## Source et extraction", "",
          "- **Cours** : ilboursa.com, téléchargement CSV « /marches/download/<CODE> » par tranches trimestrielles "
          "(cours bruts, non ajustés), `src/collect_ilboursa.py`. Tunindex = code PX1.",
          f"- **Date d'extraction** : {date.today().isoformat()} ; période demandée : {START.date()} → {END.date()} ; "
          f"Tunindex : {index.index[0].date()} → {index.index[-1].date()} ({len(index)} séances).",
          "- **Opérations sur titres** : CMF (cmf.tn) — résolutions d'AG (2023 et 2026 disponibles en ligne), "
          "« augmentations de capital réalisées », « informations des sociétés » (communiqués 2023-2026) ; "
          "à défaut, pour les dividendes 2023-2025 : montant de la fiche société ilboursa + date détectée sur les cours.",
          "- **Contrôle externe** : exports investing.com de BIAT, SFBT et du Tunindex.",
          "- Collecte : robots.txt vérifié, pauses de 3 à 5 s (ilboursa) et 10,5 s (CMF, Crawl-delay 10), cache local.",
          "",
          "### Avertissement : ouvertures ilboursa non fiables",
          "Les cours d'ouverture ilboursa sont égaux à la clôture du jour dans 35 à 60 % des séances et ne "
          "concordent avec investing.com que dans 34 % (BIAT) à 44 % (SFBT) des cas, contre 99 % pour les clôtures. "
          "Ils ne sont ni utilisés ni livrés : détection, ajustement et prix finaux reposent uniquement sur les clôtures.",
          "",
          "## Résumé", "",
          f"- Titres de l'univers : {len(st)} ; retenus : **{len(kept)}** ; exclus : {len(excl)}.",
          f"- Lignes de `prix_ajustes.csv` : {len(prices)}.",
          f"- Ajustements appliqués : {len(applied)} ; variations > 25 % listées : {len(big)}.",
          f"- Dividendes ajustés : {n_div_exact + n_div_approx} (date exacte : {n_div_exact} — CMF {n_div_cmf}, "
          f"ilboursa + détection {n_div_exact - n_div_cmf} ; date approximative : {n_div_approx}, dont "
          f"{n_div_approx - n_div_approx0} avec plusieurs séances compatibles et {n_div_approx0} sans séance "
          f"compatible mais baisse observée ≥ 50 % de l'attendue) ; "
          f"non ajustés : {len(no_match)} ; biais résiduel médian estimé sur le rendement annualisé des titres concernés : "
          f"{bias_median}.",
          "- Filtres : historique ≥ 2 ans (première à dernière cotation) ; liquidité ≥ 60 % des séances du "
          "Tunindex (depuis la première cotation) avec volume > 0.",
          "", "## Contrôle BIAT / SFBT contre investing.com", "",
          "investing.com ajuste les attributions gratuites mais pas les dividendes : le cours ilboursa ajusté "
          "des seules attributions doit coïncider avec investing.com.", ""]
    L += investing_control(prices)
    L += ["", "## Jours manquants par titre", "",
          "Séances du Tunindex, de la première cotation du titre à la fin, sans cours du titre.", "",
          md_table(missing), "",
          "## Titres exclus et motif", "", md_table(excl[["ticker", "nom", "motif"]]), "",
          "## Ajustements appliqués", "",
          "Facteur appliqué à tous les cours antérieurs à la date. Source de la confirmation indiquée pour chaque ligne.", "",
          md_table(applied[["ticker", "date", "type", "valeur", "facteur", "calcul", "source", "verification"]]
                   if len(applied) else applied), ""]
    if len(cmf_mismatch):
        L += ["### Dividendes CMF ajustés à la date du document, non visibles dans les cours", "",
              md_table(cmf_mismatch[["ticker", "date", "valeur", "verification"]]), ""]
    # --- erreurs de données, ajustements déjà faits par ilboursa, cas lus dans les documents
    corr = read(DATA / "corrections_appliquees.csv")
    scan = read(DATA / "prix_decimales_anormales.csv")
    ilb = read(DATA / "ajustements_ilboursa.csv")
    cas = read(DATA / "cas_documentaires.csv")
    neut = read(DATA / "seances_neutralisees.csv")
    if len(neut):
        flagged = prices[prices["seance_neutralisee"].eq(1)]
        L += ["## Séances neutralisées (baisse gardée, à tester)", "",
              "Baisses proches d'une parité réaliste, sans document ni contrôle investing.com possible. Le cours "
              "n'est pas ajusté (la baisse est traitée comme réelle) ; dans `prix_ajustes.csv`, la colonne "
              "`seance_neutralisee` vaut 1 ce jour-là : pour la variante « neutralisée », mettre à zéro (ou exclure) "
              "le rendement de cette séance. Si un CSV investing.com quotidien (Tjari.csv, Cil.csv, Cell.csv) est "
              "ajouté, refaire le contrôle rapport ilboursa/investing comme pour BIAT/SFBT.", "",
              md_table(neut), "",
              f"Séances marquées dans `prix_ajustes.csv` : {len(flagged)} "
              f"({', '.join(f'{t} {d}' for t, d in zip(flagged['ticker'], flagged['date']))}).", ""]
    L += ["## Erreurs de données corrigées", "",
          "Séances supprimées avant toute étape (`data/corrections_donnees.csv`, appliqué par "
          "`src/corriger_donnees.py` dans `data/cours_corriges/`, cours ilboursa d'origine conservés dans "
          "`mes_exports/`). Le titre est traité comme non coté ce jour-là.", "",
          md_table(corr) if len(corr) else "_Aucune._", "",
          "Recherche des clôtures à plus de 2 décimales (la cotation se fait au centime) dans tous les titres :", ""]
    if len(scan):
        g = scan.groupby(["ticker", "classement"]).agg(seances=("date", "size"), premiere=("date", "min"),
                                                         derniere=("date", "max")).reset_index()
        L += [md_table(g), "",
              "Une valeur isolée est une erreur (corrigée ci-dessus). Un bloc correspond à des cours déjà "
              "ajustés par ilboursa (section suivante) : ce ne sont pas des erreurs.", ""]
    if len(ilb):
        L += ["## Opérations déjà ajustées par ilboursa", "",
              "Pour ces titres, ilboursa a ajusté rétroactivement ses cours « bruts » (cours réels divisés par "
              "un facteur F, d'où des clôtures à pleine précision) : pas de saut le jour de l'opération, donc "
              "**aucun ajustement supplémentaire** ici. Les dividendes publiés (fiche ilboursa, CMF) restent par "
              "ancienne action : un dividende détaché avant l'opération est converti en D / F avant de "
              "calculer le facteur (colonne « calcul » des ajustements). Facteurs retrouvés en ramenant 100 % "
              "des cours du bloc sur la grille du centime (`src/ajustements_ilboursa.py`).", "",
              md_table(ilb[["ticker", "debut_bloc", "fin_bloc", "date_effet", "facteur_operation_libelle",
                            "facteur_cumule_libelle", "part_seances_sur_grille"]]), "",
              "Lecture : STAR 4,333 ≈ 13/3, cohérent avec le passage de 2 307 693 à 10 000 003 actions "
              "(résolutions CMF 2026) ; MPBS ×2 = la division annoncée en avril 2025 ; AB 8/7 = attribution "
              "1 pour 7, AB 1,153744 (non simple) = probablement une augmentation en numéraire (TERP) ; "
              "SAH 36/35 = attribution 1 pour 35 ; SMART 7/6 = attribution 1 pour 6 ; SMART 1,29358 et "
              "DH ×2 : nature non documentée, facteur mesuré.", ""]
    if len(cas):
        c = cas.copy()
        yrs = (pd.Timestamp(END) - pd.Timestamp(START)).days / 365.25
        c["biais_rendement_annualise_si_operation_reelle"] = [
            "" if s.startswith("résolu") else f"{(1 / float(f)) ** (1 / yrs) - 1:+.2%}"
            for s, f in zip(c["statut"], c["facteur_si_operation"])]
        L += ["## Cas lus dans les documents", "",
              "Impact si l'opération est réelle mais non ajustée : la baisse reste comme une perte d'un jour ; "
              "le rendement annualisé du titre sur la période est sous-estimé d'environ "
              "(1/facteur)^(1/années) − 1.", "",
              md_table(c[["ticker", "date", "variation_cloture", "statut", "facteur_si_operation",
                          "biais_rendement_annualise_si_operation_reelle", "documents_lus", "conclusion"]]), ""]
    L += ["## Opérations non ajustées", "",
          "### Opérations documentées mais incomplètes", "",
          md_table(skipped[["ticker", "date", "type", "valeur", "motif", "document"]] if len(skipped) else skipped), ""]
    if len(divs):
        rule = ("Règle : montant de l'exercice N (fiche ilboursa) ; séance d'avril à août de N+1 dont la baisse "
                "clôture à clôture est la plus proche de D/P(J−1), avec Tunindex < 2 %. Séance compatible = écart "
                "≤ 1,5 point. Une seule séance compatible : date exacte ; plusieurs : séance d'écart minimal, date "
                "approximative ; aucune : non ajusté.")
        if len(approx):
            L += ["### Dividendes ilboursa ajustés à une date approximative (plusieurs séances compatibles)", "",
                  rule, "",
                  md_table(approx[["ticker", "exercice", "montant", "date_detachement", "baisse_observee",
                                   "baisse_attendue", "ecart_points", "nb_seances_compatibles", "retenu"]]), ""]
        L += ["### Dividendes ilboursa non ajustés (aucune séance compatible)", "", rule, "",
              md_table(no_match[["ticker", "exercice", "montant", "meilleure_seance", "baisse_observee",
                                 "baisse_attendue", "ecart_points", "cours_veille_estime", "D_sur_P", "retenu",
                                 "a_verifier", "statut"]]) if len(no_match) else "_Aucun._", "",
              "### Biais estimé par titre (dividendes non ajustés)", "",
              "Biais sur le rendement annualisé = somme des D / P(veille estimée) des dividendes non ajustés, "
              "divisée par le nombre d'années de cotation du titre sur la période. P(veille estimée) = clôture "
              "précédant la séance la plus proche ; à défaut, clôture médiane d'avril à août de l'année de paiement. "
              "Les rendements calculés sur `close_ajuste` sont sous-estimés d'environ ce montant. Un D/P > 10 % "
              "est invraisemblable (montant ilboursa ou cours à vérifier) : ligne signalée « a_verifier ».", "",
              md_table(bias) if len(bias) else "_Aucun._", ""]
        nf = divs[divs["statut"].str.startswith("fiche") | divs["statut"].str.startswith("cours")]
        if len(nf):
            L += ["### Dividendes non vérifiables (fiche ou cours indisponibles)", "",
                  md_table(nf[["ticker", "annee_paiement", "statut"]]), ""]
    if len(docs):
        L += ["### Documents CMF à relire (champ illisible ou ambigu)", "", md_table(docs), ""]
    L += ["## Sauts très suspects sans confirmation (à vérifier)", "",
          "Baisse ≥ 8 %, à moins de 1 % d'une parité réaliste, Tunindex < 2 %, pas de rebond le lendemain "
          "(moins de la moitié de la baisse reprise), cours ≥ 1 DT. Recherche d'avis dans les augmentations de "
          "capital réalisées du CMF et dans les actualités ilboursa du titre. **Non ajustés.**", "",
          md_table(to_check[["ticker", "date", "variation_cloture", "parite_la_plus_proche", "libelle_parite",
                             "part_baisse_reprise_lendemain", "statut", "avis_cmf_augmentation", "actus_ilboursa"]]
                   if len(to_check) else to_check), ""]
    if len(sauts):
        n_small = int(sauts["petit_cours"].eq("oui").sum())
        L += [f"Sauts détectés au total (baisse ≥ 4 %, Tunindex < 2 %) : {len(sauts)}, dont "
              f"{int(sauts['confirme'].ne('non').sum())} confirmés et {n_small} sur des titres à moins de 1 DT "
              "(pas de cotation 0,01 : rapports « exacts » artificiels, jamais traités comme attributions). "
              "Détail : `data/sauts_suspects.csv`.", ""]
    L += ["## Variations journalières > 25 % après ajustement", "",
          md_table(big) if len(big) else "_Aucune._", ""]
    (FINAL / "rapport_qualite.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
