"""Convertit des exports de cotations vers le format du pipeline :
prices_long.csv (date, ticker, close, volume) et tunindex.csv (date, close).

Deux formats sont reconnus automatiquement :
  - ilboursa      : symbole, date, ouverture, haut, bas, cloture, volume   (le symbole est dans le fichier)
  - investing.com : Date, Price, Open, High, Low, Vol., Change %            (pas de symbole : le NOM DU FICHIER
                    sert de code, par exemple BIAT.csv ou TUNINDEX.csv)

Usage :
    python src/convert_ilboursa.py --in-dir mes_exports --out-dir data/raw

Le fichier de l'indice doit s'appeler TUNINDEX (ou avoir le symbole TUNINDEX).
"""
import argparse
import glob
import os

import pandas as pd


def num_fr(s):
    """ilboursa : décimale en point (ou virgule), pas de séparateur de milliers."""
    return pd.to_numeric(s.astype(str).str.replace(r"\s", "", regex=True).str.replace(",", "."), errors="coerce")


def num_en(s):
    """investing.com : virgule = séparateur de milliers, point = décimale."""
    return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")


def vol_en(s):
    """Volumes investing.com : '30.96K', '1.02M', vide si inconnu."""
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}

    def one(x):
        x = str(x).strip().replace(",", "")
        if x in ("", "nan", "-"):
            return float("nan")
        return float(x[:-1]) * mult[x[-1].upper()] if x[-1].upper() in mult else float(x)

    return s.map(one)


def parse_dates(s, path=""):
    """Dates investing.com : jj/mm/aaaa OU mm/jj/aaaa selon la région du compte. On teste les deux formats.
    Les fichiers sont triés par date : le bon format est celui qui donne une série monotone."""
    s = s.astype(str).str.strip()
    best, best_score = None, None
    for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
        d = pd.to_datetime(s, format=fmt, errors="coerce")
        diffs = d.diff().dropna()
        violations = int((diffs.dt.days > 0).sum()) if (diffs.dt.days < 0).sum() >= (diffs.dt.days > 0).sum() else int((diffs.dt.days < 0).sum())
        score = (int(d.isna().sum()), violations)
        if best_score is None or score < best_score:
            best, best_score, best_fmt = d, score, fmt
    if best_score[0] or best_score[1]:
        print(f"ATTENTION {os.path.basename(path)} : dates ambiguës ou désordonnées (NaT={best_score[0]}, ruptures={best_score[1]}), vérifiez le fichier")
    print(f"  {os.path.basename(path)} : format de date détecté {best_fmt}")
    return best


def read_any(path):
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")


def load_file(path):
    raw = read_any(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "price" in raw.columns:                                   # investing.com
        tk = os.path.splitext(os.path.basename(path))[0].upper()
        return pd.DataFrame({
            "ticker": tk,
            "date": parse_dates(raw["date"], path),
            "open": num_en(raw["open"]), "close": num_en(raw["price"]), "volume": vol_en(raw["vol."]),
        })
    return pd.DataFrame({                                         # ilboursa
        "ticker": raw["symbole"].astype(str).str.strip().str.upper(),
        "date": pd.to_datetime(raw["date"], dayfirst=True, errors="coerce"),
        "open": num_fr(raw["ouverture"]), "close": num_fr(raw["cloture"]), "volume": num_fr(raw["volume"]),
    })


def main(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(f"{in_dir}/*.csv") + glob.glob(f"{in_dir}/*.xlsx") + glob.glob(f"{in_dir}/*.xls"))
    df = pd.concat([load_file(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["date", "close"]).drop_duplicates(["ticker", "date"]).sort_values(["ticker", "date"])

    # Contrôle : ouverture == clôture sur (presque) toutes les lignes = export suspect
    same = (df["open"] == df["close"]).groupby(df["ticker"]).mean()
    for tk, r in same.items():
        if r > 0.9:
            print(f"ATTENTION {tk} : ouverture = clôture sur {r:.0%} des lignes, vérifiez la colonne de clôture sur le site")

    idx = df[df["ticker"] == "TUNINDEX"]
    idx[["date", "close"]].to_csv(f"{out_dir}/tunindex.csv", index=False)
    df[df["ticker"] != "TUNINDEX"][["date", "ticker", "close", "volume"]].to_csv(f"{out_dir}/prices_long.csv", index=False)
    print(f"{df['ticker'].nunique() - (len(idx) > 0)} titres, {len(df)} lignes, "
          f"du {df['date'].min().date()} au {df['date'].max().date()}")
    if len(idx) == 0:
        print("ATTENTION : aucune ligne TUNINDEX, le bêta ne pourra pas être calculé")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", default="data/raw")
    a = ap.parse_args()
    main(a.in_dir, a.out_dir)
