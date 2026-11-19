# BVMT : jeu de données de cours ajustés (2023-2026)

But : construire, pour les sociétés cotées à la Bourse de Tunis, des cours de clôture journaliers ajustés
des opérations sur titres (dividendes, attributions gratuites, divisions, augmentations de capital), avec
un rapport qualité. Les rendements et indicateurs sont calculés ensuite, hors de ce dépôt
(voir `README_phase2_modeles.md` pour la suite du projet).

## Résultats (`data/final/`)

- `prix_ajustes.csv` : date, ticker, close_brut, close_ajuste, volume, seance_neutralisee
- `tunindex.csv` : clôtures du Tunindex (code ilboursa PX1)
- `titres_retenus.csv` : titres passant les filtres (historique ≥ 2 ans, ≥ 60 % de séances avec volume)
- `rapport_qualite.md` : sources, jours manquants, exclusions, ajustements, cas non résolus

Fichiers de décision (modifiables à la main, relus par `build_final.py`) dans `data/` :
`corrections_donnees.csv`, `operations_validees.csv`, `seances_neutralisees.csv`, `titres_sans_cotation.csv`,
`cas_documentaires.csv` ; tables produites : `operations_sur_titres.csv`, `dividendes_ilboursa.csv`,
`ajustements_ilboursa.csv`, `sauts_suspects.csv`, `sauts_a_verifier.csv`.

## Relancer

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows : .venv\Scripts\pip

# 1. Collecte (requêtes espacées, robots.txt respecté ; plusieurs heures).
#    L'univers univers_BVMT_ilboursa_2026-11-18.csv est attendu dans le dossier parent.
python src/collect_ilboursa.py --phase full          # cours ilboursa -> mes_exports/
python src/collect_ost_cmf.py --step all             # avis CMF -> data/ost_cmf/
python src/dividendes_ilboursa.py --step fetch       # fiches société ilboursa (dividendes)

# 2. Préparation et détection
python src/corriger_donnees.py                       # erreurs de données -> data/cours_corriges/
python src/ajustements_ilboursa.py                   # opérations déjà ajustées par ilboursa
python src/extract_ost.py
python src/detect_sauts.py
python src/dividendes_ilboursa.py --step detect
python src/extract_ost.py                            # fusion des dividendes ilboursa
python src/detect_sauts.py
python src/verifier_sauts.py                         # --no-fetch pour n'utiliser que le cache

# 3. Jeu final
python src/build_final.py                            # -> data/final/
```

Les caches (cours bruts, PDF, pages web) ne sont pas versionnés : la collecte les recrée.
