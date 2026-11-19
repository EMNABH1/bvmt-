"""Collecte respectueuse des historiques ilboursa, par tranches trimestrielles."""
import argparse
import csv
from datetime import date, datetime, timedelta
import io
import logging
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin
import urllib.robotparser

from bs4 import BeautifulSoup
import requests


BASE_URL = "https://www.ilboursa.com"
LIST_URL = f"{BASE_URL}/marches/aaz"
DOWNLOAD_PAGE = f"{BASE_URL}/marches/download/{{code}}"
DOWNLOAD_URL = f"{BASE_URL}/marches/download/{{code}}"
USER_AGENT = "BVMT-Phase2-student-project/1.0 (academic data collection)"
START_DATE = date(2023, 1, 1)
END_DATE = date(2026, 11, 19)
PAUSE_RANGE_SECONDS = (3.0, 5.0)
VALIDATION_CODES = ("BIAT", "SFBT", "PX1")
EXPECTED_COLUMNS = {"date", "cloture"}
COMPARISON_TOLERANCE = 0.01
MIN_OVERLAP_ROWS = 20
MIN_COVERAGE = 0.95
# Noms de l'univers absents (ou libellés autrement) sur la page A à Z, vérifiés à la main.
# TUNINVEST : libellé « TUNINVEST - SICAR » sur ilboursa, page cotation_TINV vérifiée le 2026-11-19.
MANUAL_CODES = {"TUNINVEST - SICAF": "TINV"}
NO_DATA_MARKER = "Pas de données disponibles pour les dates choisies"
EMPTY_CSV_HEADER = b"symbole;date;ouverture;haut;bas;cloture;volume\r\n"
CHALLENGE_MARKERS = (
    "cf-chl-",
    "cloudflare",
    "just a moment",
    "attention required",
    "verify you are human",
    "/cdn-cgi/challenge-platform/",
)
ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent


class CollectionStopped(RuntimeError):
    """Raised when collection must stop without retrying."""


class PoliteSession:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.request_made = False
        self.robots_parser = None

    def request(self, method, url, **kwargs):
        if self.request_made:
            time.sleep(random.uniform(*PAUSE_RANGE_SECONDS))
        self.request_made = True
        for attempt in range(3):
            try:
                response = self.session.request(
                    method, url, timeout=30, allow_redirects=False, **kwargs
                )
                break
            except (requests.ConnectionError, requests.Timeout) as error:
                # Coupure réseau ponctuelle : pause longue (60 s puis 120 s), 2 reprises au plus.
                if attempt == 2:
                    raise CollectionStopped(f"Arrêt : connexion impossible après 3 essais sur {url} ({error})")
                logging.warning("Coupure réseau sur %s (%s) : nouvel essai dans %d s", url, error, 60 * (attempt + 1))
                time.sleep(60 * (attempt + 1))
        self.check_response(response)
        return response

    @staticmethod
    def check_response(response):
        if response.status_code in (403, 503):
            raise CollectionStopped(
                f"Arrêt immédiat : HTTP {response.status_code} sur {response.url}"
            )
        excerpt = response.content[:200_000].decode("utf-8", errors="replace").lower()
        if any(marker in excerpt for marker in CHALLENGE_MARKERS):
            raise CollectionStopped(
                f"Arrêt immédiat : challenge Cloudflare détecté sur {response.url}"
            )
        if 300 <= response.status_code < 400:
            raise CollectionStopped(
                f"Arrêt : redirection HTTP {response.status_code} non suivie sur {response.url}"
            )
        response.raise_for_status()

    def close(self):
        self.session.close()


def setup_logging(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
        force=True,
    )


def allowed_by_robots(client, url):
    if client.robots_parser is None:
        robots_url = f"{BASE_URL}/robots.txt"
        response = client.request("GET", robots_url)
        client.robots_parser = urllib.robotparser.RobotFileParser()
        client.robots_parser.set_url(robots_url)
        client.robots_parser.parse(response.text.splitlines())
    if not client.robots_parser.can_fetch(USER_AGENT, url):
        raise CollectionStopped(f"robots.txt interdit l'accès à {url}")


def normalize_name(value):
    return " ".join(value.strip().upper().split())


def load_universe(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        if not reader.fieldnames or "nom_ilboursa" not in reader.fieldnames:
            raise ValueError(f"Colonne nom_ilboursa introuvable dans {path}")
        return [row["nom_ilboursa"].strip() for row in reader if row.get("nom_ilboursa")]


def discover_codes(client):
    response = client.request("GET", LIST_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    found = {}
    for link in soup.find_all("a", href=True):
        match = re.search(r"(?:^|/)cotation_([A-Za-z0-9]+)$", link["href"].strip())
        name = link.get_text(" ", strip=True)
        if match and name:
            found[normalize_name(name)] = match.group(1).upper()
    if not found:
        raise RuntimeError("Aucune correspondance nom/code trouvée sur la page A à Z.")
    return found


def save_code_map(universe_path, output_path, found):
    names = load_universe(universe_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing = []
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("nom", "code"))
        for name in names:
            code = found.get(normalize_name(name), "") or MANUAL_CODES.get(name, "")
            writer.writerow((name, code))
            if not code:
                missing.append(name)
                logging.error("Code ilboursa introuvable : %s", name)
    for name in missing:
        print(f"Code introuvable : {name}")
    print(f"Correspondances enregistrées : {output_path} ({len(names) - len(missing)}/{len(names)})")
    result = {name: found[normalize_name(name)] for name in names if normalize_name(name) in found}
    for name in names:
        if name not in result and name in MANUAL_CODES:
            result[name] = MANUAL_CODES[name]
    return result


def read_code_map(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {
            row["nom"]: row["code"].strip().upper()
            for row in csv.DictReader(file)
            if row.get("code", "").strip()
        }


def quarter_ranges(start=START_DATE, end=END_DATE):
    cursor = start
    while cursor <= end:
        next_month_index = cursor.year * 12 + cursor.month - 1 + 3
        next_start = date(next_month_index // 12, next_month_index % 12 + 1, 1)
        tranche_end = min(end, next_start - timedelta(days=1))
        yield cursor, tranche_end
        cursor = tranche_end + timedelta(days=1)


def parse_csv_rows(content):
    text = content.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    if not reader.fieldnames:
        raise ValueError("CSV sans en-tête.")
    fields = {field.strip().lower() for field in reader.fieldnames if field}
    if not EXPECTED_COLUMNS.issubset(fields):
        raise ValueError(f"En-tête CSV inattendu : {reader.fieldnames}")
    rows = []
    for row in reader:
        normalized = {key.strip().lower(): (value or "").strip() for key, value in row.items() if key}
        if normalized.get("date") and normalized.get("cloture"):
            rows.append(normalized)
    return [field.strip().lower() for field in reader.fieldnames], rows


def tranche_path(directory, code, start, end):
    return directory / code / f"{start.isoformat()}_{end.isoformat()}.csv"


def fetch_tranche(client, code, start, end, output_path):
    if output_path.exists():
        try:
            parse_csv_rows(output_path.read_bytes())
        except (UnicodeDecodeError, csv.Error, ValueError) as error:
            logging.error("Tranche existante invalide, non remplacée %s : %s", output_path, error)
            raise RuntimeError(f"Tranche existante invalide : {output_path}") from error
        logging.info("Tranche déjà présente, ignorée : %s", output_path)
        return

    page_url = DOWNLOAD_PAGE.format(code=code)
    allowed_by_robots(client, page_url)
    page = client.request("GET", page_url, headers={"Referer": f"{BASE_URL}/"})
    soup = BeautifulSoup(page.text, "html.parser")
    token = soup.find("input", attrs={"name": "__RequestVerificationToken"})
    if token is None or not token.get("value"):
        raise RuntimeError(f"Jeton __RequestVerificationToken introuvable pour {code}")
    form = token.find_parent("form")
    if form is None:
        raise RuntimeError(f"Formulaire du jeton introuvable pour {code}")

    fields = {
        field["name"]: field.get("value", "")
        for field in form.find_all("input", attrs={"type": "hidden", "name": True})
    }
    fields.update({"dtFrom": start.isoformat(), "dtTo": end.isoformat()})
    post_url = urljoin(BASE_URL, form.get("action") or DOWNLOAD_URL.format(code=code))
    result = client.request(
        "POST",
        post_url,
        data=fields,
        headers={"Referer": page.url, "Origin": BASE_URL},
    )
    content = result.content
    if content[:200].lstrip().lower().startswith(b"<!doctype html") and NO_DATA_MARKER in " ".join(
        BeautifulSoup(result.text, "html.parser").get_text(" ").split()
    ):
        # Aucune cotation sur la période (titre suspendu, introduit ou radié) : tranche vide.
        logging.warning("Pas de données ilboursa pour %s %s–%s : tranche vide", code, start, end)
        print(f"{code}: {start} → {end} (aucune donnée)")
        content = EMPTY_CSV_HEADER
    parse_csv_rows(content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(output_path)
    logging.info("Tranche téléchargée %s %s–%s (HTTP %s)", code, start, end, result.status_code)
    print(f"{code}: {start} → {end} (HTTP {result.status_code})")


def parse_date(value):
    value = value.strip()
    for pattern in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"Date ilboursa invalide : {value}")


def merge_code(code, chunk_directory, output_directory):
    rows_by_date = {}
    columns = None
    for start, end in quarter_ranges():
        path = tranche_path(chunk_directory, code, start, end)
        if not path.exists():
            return False
        current_columns, rows = parse_csv_rows(path.read_bytes())
        if columns is None:
            columns = current_columns
        elif current_columns != columns:
            raise ValueError(f"En-têtes différents dans les tranches de {code}")
        for row in rows:
            rows_by_date[parse_date(row["date"])] = row

    if columns is None:
        raise RuntimeError(f"Aucune tranche disponible pour {code}")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{code}.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows_by_date[key] for key in sorted(rows_by_date))
    print(f"Fusionné : {output_path} ({len(rows_by_date)} séances)")
    return True


def collect_codes(client, codes, chunk_directory, export_directory):
    for code in codes:
        for start, end in quarter_ranges():
            path = tranche_path(chunk_directory, code, start, end)
            fetch_tranche(client, code, start, end, path)
        merge_code(code, chunk_directory, export_directory)


def parse_price(value, decimal_comma):
    value = value.strip().replace("\u00a0", "").replace(" ", "")
    if decimal_comma:
        if "," in value and "." in value:
            if value.rfind(",") > value.rfind("."):
                value = value.replace(".", "").replace(",", ".")
            else:
                value = value.replace(",", "")
        else:
            value = value.replace(",", ".")
    else:
        value = value.replace(",", "")
    return float(value)


def read_ilboursa_prices(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        return {
            parse_date(row["date"]): parse_price(row["cloture"], decimal_comma=True)
            for row in reader
            if row.get("date") and row.get("cloture")
        }


def read_investing_prices(path):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        prices = {}
        for row in reader:
            if not row.get("Date") or not row.get("Price"):
                continue
            day = datetime.strptime(row["Date"].strip(), "%m/%d/%Y").date()
            if day >= START_DATE:
                prices[day] = parse_price(row["Price"], decimal_comma=False)
        return prices


def compare_validation(export_directory, reference_directory, coverage_only=False):
    """Compare les clôtures ilboursa aux exports investing.com.

    coverage_only=True : les écarts de prix ne bloquent pas (ilboursa publie des cours
    bruts, investing.com des cours ajustés des opérations sur titres) ; seule la
    couverture des dates est exigée.
    """
    references = {
        "BIAT": reference_directory / "Biat.csv.csv",
        "SFBT": reference_directory / "Sfbt.csv.csv",
        "PX1": reference_directory / "Tunidex.csv.csv",
    }
    all_concordant = True
    for code, reference_path in references.items():
        exported_path = export_directory / f"{code}.csv"
        if not reference_path.exists() or not exported_path.exists():
            print(f"Comparaison {code}: fichier manquant, validation refusée.")
            logging.error("Fichier de comparaison manquant pour %s", code)
            all_concordant = False
            continue
        actual = read_ilboursa_prices(exported_path)
        expected = read_investing_prices(reference_path)
        shared_days = sorted(actual.keys() & expected.keys())
        matches = [
            day for day in shared_days
            if abs(actual[day] - expected[day]) <= COMPARISON_TOLERANCE
        ]
        mismatches = [day for day in shared_days if day not in set(matches)]
        coverage = len(shared_days) / len(expected) if expected else 0.0
        concordant = (
            len(shared_days) >= MIN_OVERLAP_ROWS
            and (coverage_only or not mismatches)
            and coverage >= MIN_COVERAGE
        )
        if coverage_only and mismatches:
            logging.warning(
                "Écarts %s attribués aux opérations sur titres (cours bruts ilboursa) : %d jours",
                code, len(mismatches),
            )
        print(
            f"Comparaison {code}: {len(matches)}/{len(shared_days)} clôtures concordantes "
            f"(±{COMPARISON_TOLERANCE:.2f}), couverture {coverage:.1%}, "
            f"{len(mismatches)} écarts, {len(expected.keys() - actual.keys())} dates manquantes."
        )
        if mismatches:
            for day in mismatches[:5]:
                print(f"  {day}: ilboursa={actual[day]:g}, Investing.com={expected[day]:g}")
        logging.info(
            "Comparaison %s: %d/%d prix concordants; couverture %.3f; écarts=%d",
            code, len(matches), len(shared_days), coverage, len(mismatches)
        )
        all_concordant = all_concordant and concordant
    return all_concordant


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("validate", "full"),
        default="validate",
        help="validate: BIAT/SFBT/PX1; full: univers complet après revalidation",
    )
    parser.add_argument(
        "--universe",
        type=Path,
        default=WORKSPACE_ROOT / "univers_BVMT_ilboursa_2026-11-18.csv",
    )
    args = parser.parse_args()
    data_directory = ROOT / "data"
    chunk_directory = data_directory / "ilboursa_tranches"
    export_directory = ROOT / "mes_exports"
    code_map_path = data_directory / "codes_ilboursa.csv"
    setup_logging(ROOT / "logs" / "collecte.log")
    client = PoliteSession()

    try:
        allowed_by_robots(client, LIST_URL)
        found = discover_codes(client)
        codes_by_name = save_code_map(args.universe, code_map_path, found)

        if args.phase == "validate":
            collect_codes(client, VALIDATION_CODES, chunk_directory, export_directory)
            if compare_validation(export_directory, WORKSPACE_ROOT):
                print("Validation concordante. Relancer avec --phase full pour l'univers complet.")
            else:
                print("Validation non concordante : collecte complète non lancée.")
                logging.error("Validation non concordante; phase complète non lancée.")
                return 1
            return 0

        if not all((export_directory / f"{code}.csv").exists() for code in VALIDATION_CODES):
            print("Fichiers de validation absents : exécuter d'abord --phase validate.")
            return 1
        if not compare_validation(export_directory, WORKSPACE_ROOT, coverage_only=True):
            print("Couverture insuffisante : collecte complète non lancée.")
            logging.error("Revalidation (couverture) échouée; phase complète non lancée.")
            return 1
        universe_codes = list(dict.fromkeys(codes_by_name.values()))
        codes = list(dict.fromkeys(universe_codes + ["PX1"]))
        collect_codes(client, codes, chunk_directory, export_directory)
        print(f"Collecte terminée : {len(codes)} codes demandés.")
        return 0
    except CollectionStopped as error:
        logging.error("%s", error)
        print(error)
        return 2
    except Exception:
        logging.exception("Échec de la collecte")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
