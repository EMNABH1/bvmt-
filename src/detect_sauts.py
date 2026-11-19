"""Sauts suspects dans les cours bruts ilboursa (attributions gratuites / divisions probables).

Critère : séance t où la clôture baisse de plus de 4 % par rapport à la clôture de la
séance cotée précédente, alors que le Tunindex (PX1, clôture à clôture) varie de moins
de 2 % sur le même intervalle. Seuil volontairement bas : les faux positifs sont écartés
par la confirmation.

Uniquement des clôtures : les ouvertures ilboursa ne sont pas fiables (égales à la
clôture du jour dans 35 à 60 % des séances, ≠ investing.com dans 56 à 66 % des cas).
Elles figurent à titre indicatif (colonnes ouverture, rapport_P_prec_sur_ouverture).

Pour chaque saut, rapport R = P(t-1)/P(t) sur les clôtures. Parités réalistes uniquement,
tolérance 1 % :
  - attribution n pour m : R = (m + n) / m, n = 1 et m = 1..20 ; n = 2 ou 3 et m ≤ 10
  - division k pour 1    : R = k, k ∈ {2, 4, 5, 10}

Titres à petit cours (P(t-1) < 1 DT) : le pas de cotation de 0,01 DT fabrique des rapports
« exacts » (0,75 -> 0,60 = 5/4) ; ils sont signalés et jamais comptés comme attributions.

Colonne « confirme » :
  - « CMF »       : document CMF du même titre (communiqué « Informations des sociétés » ou
                    opération de data/operations_sur_titres.csv) daté à ± 10 séances du saut ;
  - « investing » : titre disposant d'un export investing.com (BIAT, SFBT) dont le rapport
                    ilboursa/investing vaut la parité avant le saut (médiane des 5 séances
                    précédentes) et 1 après (médiane des 5 séances à partir du saut), ± 0,5 % ;
  - « CMF+investing » si les deux ; « non » sinon.
L'ajustement des prix n'utilise QUE les sauts confirmés (CMF et/ou investing).

Sortie : data/sauts_suspects.csv
"""
from fractions import Fraction
from math import gcd
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_ilboursa import read_investing_prices  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "data" / "cours_corriges"  # mes_exports + data/corrections_donnees.csv (src/corriger_donnees.py)
OUT = ROOT / "data" / "sauts_suspects.csv"
INFOS = ROOT / "data" / "ost_cmf" / "index_infos.csv"
OPS = ROOT / "data" / "operations_sur_titres.csv"
DROP_THRESHOLD = -0.04
INDEX_THRESHOLD = 0.02
PARITY_TOLERANCE = 0.01
SMALL_PRICE = 1.0
CONFIRM_WINDOW = 10  # séances
INVESTING_FILES = {"BIAT": ROOT.parent / "Biat.csv.csv", "SFBT": ROOT.parent / "Sfbt.csv.csv"}
INVESTING_WINDOW = 5
INVESTING_TOLERANCE = 0.005


def candidate_parities():
    pairs = [(1, m) for m in range(1, 21)] + [(n, m) for n in (2, 3) for m in range(1, 11) if gcd(n, m) == 1]
    out = {}
    for n, m in pairs:
        out.setdefault(Fraction(m + n, m), f"attribution {n} pour {m}")
    for k in (2, 4, 5, 10):
        f = Fraction(k, 1)
        out[f] = (out[f] + " / " if f in out else "") + f"division {k} pour 1"
    return out


PARITIES = candidate_parities()


def nearest_parity(ratio):
    f, label = min(PARITIES.items(), key=lambda kv: abs(ratio / float(kv[0]) - 1))
    return f, label, ratio / float(f) - 1


def load(code):
    df = pd.read_csv(EXPORTS / f"{code}.csv", sep=";", encoding="utf-8-sig", dtype=str)
    if df.empty:
        return pd.DataFrame(columns=["ouverture", "cloture"], dtype=float)
    out = pd.DataFrame({c: pd.to_numeric(df[c].str.replace(",", ".").str.replace(" ", ""), errors="coerce")
                        for c in ("ouverture", "cloture")})
    out.index = pd.to_datetime(df["date"], format="%d/%m/%Y")
    return out.dropna(subset=["cloture"]).sort_index()


def cmf_documents():
    """(ticker, date, type, référence) de tous les documents CMF disponibles."""
    docs = []
    if INFOS.exists():
        infos = pd.read_csv(INFOS, dtype=str).fillna("")
        for _, r in infos.iterrows():
            if r["ticker"]:
                docs.append((r["ticker"], pd.Timestamp(r["date_publication"]), "communiqué CMF",
                             f"{r['titre']} <{r['page']}>", None))
    if OPS.exists():
        ops = pd.read_csv(OPS, dtype=str).fillna("")
        # Seuls les documents CMF confirment : pas les dividendes « ilboursa + détection »,
        # dont la date vient justement des cours.
        ops = ops[ops["document_source"].str.contains("cmf.tn", regex=False)]
        for _, r in ops.iterrows():
            for col in ("date_detachement", "date"):
                if r.get(col):
                    docs.append((r["ticker"], pd.Timestamp(r[col]), r["type"],
                                 f"{r['type']} {r['montant_ou_parite']} <{r['document_source']}>",
                                 r["montant_ou_parite"] if r["type"] == "attribution_gratuite" and r["montant_ou_parite"] else None))
    return pd.DataFrame(docs, columns=["ticker", "date", "type", "reference", "parite"])


def investing_ratio(code, close):
    """Rapport ilboursa/investing par date (None si pas d'export investing pour ce titre)."""
    path = INVESTING_FILES.get(code)
    if path is None or not path.exists():
        return None
    inv = pd.Series(read_investing_prices(path))
    inv.index = pd.to_datetime(inv.index)
    return (close / inv).dropna().sort_index()


def investing_confirms(ratio, day):
    """(confirmé, parité, détail). La parité est lue dans le rapport ilboursa/investing
    d'avant le saut (investing.com ajuste les attributions) : le cours du jour J, qui
    peut rouvrir loin du cours de référence ajusté, n'intervient pas."""
    if ratio is None:
        return False, None, ""
    before = ratio[ratio.index < day].tail(INVESTING_WINDOW)
    after = ratio[ratio.index >= day].head(INVESTING_WINDOW)
    if len(before) < 3 or len(after) < 3:
        return False, None, ""
    b, a = before.median(), after.median()
    parity, label, err = nearest_parity(b)
    ok = abs(err) <= INVESTING_TOLERANCE and abs(a - 1) <= INVESTING_TOLERANCE and abs(b - 1) > INVESTING_TOLERANCE
    return ok, (parity, label) if ok else None, f"rapport ilboursa/investing {b:.4f} avant -> {a:.4f} après"


def session_distance(calendar, d1, d2):
    return abs(int(calendar.searchsorted(d1)) - int(calendar.searchsorted(d2)))


def main():
    index = load("PX1")["cloture"]
    calendar = index.index
    docs = cmf_documents()
    rows = []
    for path in sorted(EXPORTS.glob("*.csv")):
        code = path.stem
        if code == "PX1":
            continue
        df = load(code)
        df = df[df["cloture"] > 0]
        s, opens = df["cloture"], df["ouverture"]
        inv_ratio = investing_ratio(code, s)
        for (d0, p0), (d1, p1) in zip(s.iloc[:-1].items(), s.iloc[1:].items()):
            o1 = opens.get(d1)
            has_open = pd.notna(o1) and o1 > 0
            var = p1 / p0 - 1  # clôture à clôture
            if var >= DROP_THRESHOLD:
                continue
            i0, i1 = index.asof(d0), index.asof(d1)
            ivar = i1 / i0 - 1 if pd.notna(i0) and pd.notna(i1) else float("nan")
            if not abs(ivar) < INDEX_THRESHOLD:
                continue
            ratio = p0 / p1
            ratio_o = p0 / o1 if has_open else float("nan")
            f, label, err = nearest_parity(ratio)  # clôtures seulement
            close_to_parity = abs(err) <= PARITY_TOLERANCE
            small = p0 < SMALL_PRICE
            near = docs[docs["ticker"].eq(code)] if len(docs) else docs
            near = near[[session_distance(calendar, d, d1) <= CONFIRM_WINDOW for d in near["date"]]] if len(near) else near
            inv_ok, inv_parity, inv_detail = investing_confirms(inv_ratio, d1)
            levels = (["CMF"] if len(near) else []) + (["investing"] if inv_ok else [])
            # Parité retenue pour l'ajustement : celle du document CMF (opération extraite,
            # unique) sinon celle lue chez investing ; jamais celle du seul cours.
            doc_par = near[near["type"].eq("attribution_gratuite")]["parite"].dropna().unique() if len(near) else []
            if len(doc_par) == 1:
                n, m = map(int, doc_par[0].split(":"))
                kept_parity, parity_src = f"{m + n}/{m} (attribution {n} pour {m})", "document CMF"
            elif inv_ok:
                kept_parity, parity_src = f"{inv_parity[0].numerator}/{inv_parity[0].denominator} ({inv_parity[1]})", "investing"
            else:
                kept_parity, parity_src = "", ""
            rows.append({
                "ticker": code, "date": d1.date().isoformat(), "date_seance_precedente": d0.date().isoformat(),
                "cours_precedent": p0, "cours": p1, "variation_cloture": round(var, 4),
                "variation_tunindex": round(ivar, 4), "rapport_P_prec_sur_P": round(ratio, 4),
                "ouverture_indicative": o1, "rapport_P_prec_sur_ouverture": round(ratio_o, 4),
                "parite_la_plus_proche": f"{f.numerator}/{f.denominator}", "libelle_parite": label,
                "ecart_relatif": round(err, 4), "proche_parite_simple": "oui" if close_to_parity else "non",
                "petit_cours": "oui" if small else "non",
                "attribution_probable": "oui" if close_to_parity and not small else "non",
                "confirme": "+".join(levels) or "non",
                "controle_investing": inv_detail,
                "parite_retenue": kept_parity, "source_parite": parity_src,
                "document_confirmation": " || ".join(f"{d.date()} {t}: {ref}" for _, (d, t, ref)
                                                     in near[["date", "type", "reference"]].iterrows()),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")
    print(f"{len(out)} sauts suspects -> {OUT}")
    if len(out):
        print(out[["ticker", "date", "variation_cloture", "variation_tunindex", "rapport_P_prec_sur_P",
                   "rapport_P_prec_sur_ouverture", "parite_la_plus_proche", "ecart_relatif",
                   "petit_cours", "attribution_probable", "confirme"]].to_string(index=False))


if __name__ == "__main__":
    main()
