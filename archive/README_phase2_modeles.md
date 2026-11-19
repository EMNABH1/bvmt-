# Phase 2 : implémentation (BVMT, apprentissage non supervisé)

Couvre le travail demandé en Phase 2 du cahier des charges : collecte et préparation des données, features,
Markowitz analytique, segmentation (k-means++, PCA) et détection d'anomalies (Isolation Forest).

## Ce qu'il faut savoir avant de commencer

- **Le jeu de données BVMT n'est pas fourni.** Le cahier des charges demande que chaque équipe collecte ses
  données (scraping + note qualité, livrable de la semaine 2). `data/synthetic/` est un jeu **synthétique**
  qui sert uniquement à tester le code ; `outputs_exemple_SYNTHETIQUE/` en est le résultat. Ne jamais
  rapporter ces chiffres comme des résultats réels.
- `src/collect.py` n'a **pas été testé sur les sites réels** : il faut renseigner l'URL observée, vérifier
  `robots.txt` et les conditions d'utilisation, et valider la source avec l'encadrant.

## Données à fournir (dans `data/raw/`)

| Fichier | Colonnes | Contenu |
|---|---|---|
| `prices_long.csv` | date, ticker, close, volume | cours de clôture et volume journaliers de chaque société cotée, du 2023-01-01 à la date d'extraction |
| `tunindex.csv` | date, close | clôture journalière du Tunindex |
| `corporate_actions.csv` | ticker, date, type, value | `split` (facteur), `bonus` (actions gratuites pour 1 ancienne), `dividend` (dinars par action, date de détachement) |

Sources suggérées : site de la BVMT (historiques officiels) et ilboursa. Les opérations sur titres se relèvent dans
les avis de la BVMT et les communiqués des sociétés.

## Lancer

```bash
pip install -r requirements.txt
python src/collect.py --from-csv-dir mes_csv --out-dir data/raw     # ou scraping après configuration
python run_phase2.py --data-dir data/raw --out-dir outputs --rf 0.07
```

`--rf` : taux sans risque annuel, à documenter (par exemple TMM). Il influence le Sharpe maximal.

## Correspondance avec le cahier des charges

| Exigence Phase 2 | Où |
|---|---|
| Collecte, filtres (historique, liquidité), opérations sur titres, rapport qualité | `src/collect.py`, `src/prepare.py`, `outputs/rapport_qualite.md` |
| 7 features sur rendements log, Pearson et Spearman | `src/features.py`, `features.csv`, `correlations_*.csv` |
| Variance minimale (forme fermée), Sharpe maximal sans vente à découvert, frontière, simulation en complément | `src/markowitz.py`, `markowitz_*.csv`, `figures/frontiere_efficiente.png` |
| k-means++ : plan rendement-volatilité, toutes features, avec et sans PCA, choix de k | `src/segmentation.py`, `choix_k_*.csv`, `figures/choix_k.png` |
| Isolation Forest, contamination justifiée, comparaison avec k-means | `clusters_et_anomalies.csv`, `figures/clusters_anomalies.png` |

## Choix à justifier dans le rapport

- Le script retient k par la **silhouette** ; le cahier exige au moins **deux critères** : lisez aussi coude et Calinski-Harabasz
  (`choix_k_*.csv`) et tranchez à la main.
- Taux de contamination par défaut : 5 % (`--contamination`). À justifier et à tester (1 %, 5 %, 10 %).
- Seuils de filtrage : 500 séances d'historique (environ 2 ans) et 50 % de jours échangés, modifiables.
- Kurtosis en excès (loi normale = 0). Facteur d'annualisation : 252 séances.
- Les cours manquants sont reportés (10 jours au plus) ; l'univers est ensuite restreint aux jours où tous les titres retenus ont un cours.
- Les variations extrêmes restantes sont listées dans le rapport qualité : à vérifier à la main (opération sur titre oubliée ou vraie anomalie).
