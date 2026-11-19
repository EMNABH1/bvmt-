"""Phase 2 de bout en bout : données -> features -> Markowitz -> segmentation -> anomalies.

Exemple :  python run_phase2.py --data-dir data/raw --out-dir outputs --rf 0.07
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from prepare import prepare, write_quality_report
from features import compute_features, feature_correlations, ANN
import markowitz as mk
import segmentation as sg

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data/raw")
ap.add_argument("--out-dir", default="outputs")
ap.add_argument("--start", default="2023-01-01")
ap.add_argument("--end", default=None)
ap.add_argument("--rf", type=float, default=0.0, help="taux sans risque annuel (ex. 0.07), À DOCUMENTER")
ap.add_argument("--min-history", type=int, default=500, help="séances minimales de cotation (~2 ans)")
ap.add_argument("--min-trading-ratio", type=float, default=0.5)
ap.add_argument("--contamination", type=float, default=0.05)
ap.add_argument("--k-max", type=int, default=10)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()
os.makedirs(f"{a.out_dir}/figures", exist_ok=True)
T = lambda name: f"{a.out_dir}/{name}"

# ---------- 1. Données ----------
prices, index, status, action_log, jumps, info = prepare(a.data_dir, a.start, a.end, a.min_history, a.min_trading_ratio)
write_quality_report(T("rapport_qualite.md"), status, action_log, jumps, info)
prices.to_csv(T("prix_ajustes.csv"))
print(f"[données] {info['nb_titres_retenus']}/{info['nb_titres_depart']} titres, {info['nb_jours']} séances")

# ---------- 2. Features et corrélations ----------
feat, r = compute_features(prices, index)
feat.to_csv(T("features.csv"))
pear, spear = feature_correlations(feat)
pear.to_csv(T("correlations_pearson.csv")); spear.to_csv(T("correlations_spearman.csv"))
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
for axi, m, t in zip(ax, (pear, spear), ("Pearson", "Spearman")):
    im = axi.imshow(m.values, vmin=-1, vmax=1, cmap="coolwarm")
    axi.set_xticks(range(len(m))); axi.set_xticklabels(m.columns, rotation=45, ha="right")
    axi.set_yticks(range(len(m))); axi.set_yticklabels(m.index); axi.set_title(t)
    for i in range(len(m)):
        for j in range(len(m)):
            axi.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
fig.colorbar(im, ax=ax, shrink=0.8); fig.savefig(T("figures/correlations_features.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 3. Markowitz ----------
mu = (r.mean() * ANN).values
Sigma = (r.cov() * ANN).values
tick = list(r.columns)
w_cf = mk.min_variance_closed_form(Sigma)          # forme fermée (ventes à découvert permises)
w_mv = mk.min_variance_long_only(Sigma)
w_ms = mk.max_sharpe_long_only(mu, Sigma, a.rf)
w_eq = np.ones(len(mu)) / len(mu)
port = {"1/N": w_eq, "Variance minimale (forme fermée)": w_cf, "Variance minimale (sans vente à decouvert)": w_mv,
        "Sharpe maximal (sans vente à découvert)": w_ms}
summary = pd.DataFrame({k: mk.stats(w, mu, Sigma, a.rf) for k, w in port.items()}).T
summary["nb_titres_poids>1%"] = [int((np.abs(w) > 0.01).sum()) for w in port.values()]
summary.to_csv(T("markowitz_resume.csv"))
pd.DataFrame(port, index=tick).to_csv(T("markowitz_poids.csv"))
front = mk.efficient_frontier(mu, Sigma)
rr, vv, ss = mk.random_portfolios(mu, Sigma, a.rf, seed=a.seed)
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(vv, rr, c=ss, s=3, cmap="viridis", alpha=0.5); fig.colorbar(sc, label="Sharpe")
ax.plot(front[:, 0], front[:, 1], "r-", lw=2, label="Frontière efficiente (sans vente à découvert)")
for name, w in port.items():
    s = mk.stats(w, mu, Sigma, a.rf); ax.scatter(s["volatilite"], s["rendement"], s=70, marker="*", label=name, zorder=5)
ax.set_xlabel("Volatilité annualisée"); ax.set_ylabel("Rendement annualisé"); ax.legend(fontsize=7)
ax.set_title("Markowitz : frontière et simulation aléatoire (complément)")
fig.savefig(T("figures/frontiere_efficiente.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 4. Segmentation ----------
X_plane = sg.standardize(feat[["rendement_ann", "volatilite_ann"]])
X_all = sg.standardize(feat)
X_pca, pca = sg.pca_reduce(X_all, 0.90)
spaces = {"plan_rendement_volatilite": X_plane, "toutes_features": X_all, "toutes_features_PCA": X_pca}
labels_df = pd.DataFrame(index=feat.index)
chosen = {}
fig, axes = plt.subplots(len(spaces), 3, figsize=(14, 9))
for row, (name, X) in enumerate(spaces.items()):
    sel = sg.select_k(X, 2, a.k_max, a.seed); sel.to_csv(T(f"choix_k_{name}.csv"))
    k = int(sel["silhouette"].idxmax()); chosen[name] = k
    lab, km = sg.run_kmeans(X, k, a.seed); labels_df[f"cluster_{name}"] = lab
    axes[row, 0].plot(sel.index, sel["inertie"], "o-"); axes[row, 0].set_title(f"{name} : coude")
    axes[row, 1].plot(sel.index, sel["silhouette"], "o-", color="C1"); axes[row, 1].set_title("silhouette")
    axes[row, 2].plot(sel.index, sel["calinski_harabasz"], "o-", color="C2"); axes[row, 2].set_title("Calinski-Harabasz")
fig.tight_layout(); fig.savefig(T("figures/choix_k.png"), dpi=130); plt.close(fig)

# ---------- 5. Anomalies ----------
lab_all, km_all = sg.run_kmeans(X_all, chosen["toutes_features"], a.seed)
iso_flag, iso_score = sg.isolation_forest(X_all, a.contamination, a.seed)
km_flag, km_dist = sg.kmeans_outliers(X_all, km_all, 1 - a.contamination)
labels_df["anomalie_isolation_forest"] = iso_flag
labels_df["score_isolation_forest"] = iso_score
labels_df["isole_kmeans"] = km_flag
labels_df["distance_au_centre"] = km_dist
labels_df.join(feat).to_csv(T("clusters_et_anomalies.csv"))
both = int((iso_flag & km_flag).sum())
print(f"[anomalies] IF : {iso_flag.sum()}, k-means : {km_flag.sum()}, communs : {both}")

Z = X_pca[:, :2]
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
ax[0].scatter(feat["volatilite_ann"], feat["rendement_ann"], c=labels_df["cluster_plan_rendement_volatilite"], cmap="tab10", s=30)
ax[0].set_xlabel("Volatilité annualisée"); ax[0].set_ylabel("Rendement annualisé"); ax[0].set_title(f"k-means, plan rendement-volatilité (k={chosen['plan_rendement_volatilite']})")
ax[1].scatter(Z[:, 0], Z[:, 1], c=lab_all, cmap="tab10", s=30)
ax[1].scatter(Z[iso_flag, 0], Z[iso_flag, 1], facecolors="none", edgecolors="red", s=140, label="Isolation Forest")
for i in np.where(iso_flag)[0]:
    ax[1].annotate(feat.index[i], Z[i], fontsize=7)
ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2"); ax[1].legend(); ax[1].set_title(f"Toutes features, k={chosen['toutes_features']}, anomalies")
fig.savefig(T("figures/clusters_anomalies.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

with open(T("RESUME.md"), "w", encoding="utf-8") as f:
    f.write(f"# Résumé phase 2\n\n- Titres retenus : {info['nb_titres_retenus']} / {info['nb_titres_depart']}\n- Période : {info['debut']} au {info['fin']}\n"
            f"- Taux sans risque utilisé : {a.rf:.2%}\n- k retenu (silhouette) : {chosen}\n- PCA : {pca.n_components_} composantes (90 % de variance)\n"
            f"- Anomalies : Isolation Forest {iso_flag.sum()}, k-means {km_flag.sum()}, communes {both}\n\n## Markowitz\n\n{summary.round(4).to_markdown()}\n")
print("Terminé :", a.out_dir)
