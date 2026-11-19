"""Correspondance libellés (ilboursa, CMF) -> code ilboursa pour les sociétés de l'univers."""
import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODES = ROOT / "data" / "codes_ilboursa.csv"

# Libellés employés par le CMF, en plus des noms ilboursa de data/codes_ilboursa.csv.
EXTRA_ALIASES = {
    "AIR LIQUIDE": "AL", "AETECH": "AETEC", "ATTIJARI BANK (EX BS)": "TJARI", "BH": "BH",
    "BNA ASSURANCE": "BNASS", "BNA": "BNA", "BTE": "BTE", "BT": "BT", "GIF": "GIF", "BANQUE DE TUNISIE ET DES EMIRATS": "BTE",
    "ENNAKL": "NAKL", "LAND'OR": "LNDOR", "OTH": "OTH", "PGH": "PGH", "POULINA": "PGH",
    "PLACEMENT DE TUNISIE SICAF": "PLTU", "PLACEMENTS DE TUNISIE": "PLTU", "SAM": "SAM",
    "SCB": "SCB", "SMG": "MAG", "SNMVT": "MNP", "SPDIT SICAF": "SPDIT", "SPDIT": "SPDIT",
    "TUNINVEST SICAR": "TINV", "TUNINVEST": "TINV", "ALKIMIA": "ALKIM", "ATB": "ATB", "ATL": "ATL",
    "STB": "STB", "SOCIETE TUNISIENNE DE BANQUE": "STB", "TUNISIE LEASING": "TLS", "TLF": "TLS",
    "TELNET": "TLNET", "TAWASOL GROUP HOLDING": "TGH", "TGH": "TGH", "MONOPRIX": "MNP",
    "SOTUMAG": "MGR", "SMART": "SMART", "ONE TECH HOLDING": "OTH", "CARTHAGE CEMENT": "CC",
    "CIMENTS DE BIZERTE": "SCB", "ELECTROSTAR": "LSTR", "EURO CYCLES": "ECYCL", "CEREALIS": "CREAL",
    "HEXABYTE": "XABYT", "WIFAK": "WIFAK", "WIFAK INTERNATIONAL BANK": "WIFAK", "UNIMED": "UMED",
    "TUNIS RE": "TRE", "TUNISAIR": "TAIR", "SOTRAPIL": "STPIL", "SOTIPAPIER": "STPAP",
    "ASSURANCES MAGHREBIA VIE": "AMV", "MAGHREBIA VIE": "AMV", "BH ASSURANCE": "BHASS",
    "ATTIJARI LEASING": "TJL", "ATTIJARI BANK": "TJARI", "AMEN BANK": "AB", "UIB": "UIB",
    "UBCI": "UBCI", "BIAT": "BIAT", "SFBT": "SFBT", "DELICE HOLDING": "DH", "DELICE": "DH",
    "SANIMED": "SMD", "SOMOCER": "SOMOC", "SOTEMAIL": "SOTEM", "SOTETEL": "SOTET",
    "SOTUVER": "SOTUV", "SIPHAT": "SIPHA", "SIMPAR": "SIMPA", "ESSOUKNA": "SOKNA",
    "NEW BODY LINE": "NBL", "OFFICE PLAST": "PLAST", "CITY CARS": "CITY", "CELLCOM": "CELL",
    "TUNISIE VALEURS": "TVAL", "HANNIBAL LEASE": "HL", "BEST LEASE": "BL", "BH LEASING": "BHL",
}


def fold(text):
    """Minuscules, sans accents, espaces normalisés, apostrophes droites."""
    text = str(text).replace("’", "'").replace(" ", " ").replace("\f", " ").replace("–", "-")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text.lower()).strip()


def aliases():
    out = {}
    with CODES.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("code"):
                out[fold(row["nom"])] = row["code"].strip().upper()
    for name, code in EXTRA_ALIASES.items():
        out[fold(name)] = code
    out[fold("TUNINVEST - SICAF")] = "TINV"
    return out


ALIASES = aliases()
# Alias les plus longs d'abord : « BH ASSURANCE » avant « BH ».
_ORDERED = sorted(ALIASES.items(), key=lambda kv: -len(kv[0]))
_PATTERNS = [(re.compile(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])"), c) for a, c in _ORDERED]


def ticker_for_name(name):
    """Nom exact (après normalisation) -> code, sinon None."""
    return ALIASES.get(fold(name))


def ticker_in_text(text):
    """Premier alias (le plus long) trouvé dans un titre ; None sinon."""
    t = fold(text)
    for rx, code in _PATTERNS:
        if rx.search(t):
            return code
    return None
