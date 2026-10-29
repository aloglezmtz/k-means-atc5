"""Actividad 5 - K-Means sobre Iris con pares de características.
Uso:  python kmeans_iris.py   (requiere iris.csv en la misma carpeta)
Genera: resultados/metricas.csv, resultados/centroides.csv, resultados/asignaciones.csv
        figs/ (gráficas de grupos, codo, silueta, matriz comparativa)
"""
import os, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

VARS = ["sepal.length", "sepal.width", "petal.length", "petal.width"]
ES = {"sepal.length": "Longitud del sépalo (cm)", "sepal.width": "Ancho del sépalo (cm)",
      "petal.length": "Longitud del pétalo (cm)", "petal.width": "Ancho del pétalo (cm)"}
KS = [2, 3, 4, 5]
SEED = 42

df = pd.read_csv("iris.csv")
print(df.shape); print(df.isna().sum().sum(), "datos faltantes"); print(df.describe().round(3))
os.makedirs("resultados", exist_ok=True); os.makedirs("figs", exist_ok=True)

pares = list(itertools.combinations(VARS, 2))   # 6 pares únicos
met, cen, asg = [], [], []
modelos = {}
for a, b in pares:
    X = StandardScaler().fit_transform(df[[a, b]])          # estandarización (z-score)
    sc = StandardScaler().fit(df[[a, b]])
    for k in KS:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_)
        ari = adjusted_rand_score(df["variety"], km.labels_)  # solo para comparar, no para entrenar
        met.append(dict(par=f"{a} / {b}", k=k, inercia=km.inertia_, silueta=sil, ARI_vs_especie=ari))
        orig = sc.inverse_transform(km.cluster_centers_)
        for g in range(k):
            cen.append(dict(par=f"{a} / {b}", k=k, grupo=g, **{a: orig[g, 0], b: orig[g, 1]},
                            n_flores=int((km.labels_ == g).sum())))
        for i, l in enumerate(km.labels_):
            asg.append(dict(par=f"{a} / {b}", k=k, flor=i + 1, grupo=int(l), especie=df.variety[i]))
        modelos[(a, b, k)] = (km, sc)

M = pd.DataFrame(met); M.to_csv("resultados/metricas.csv", index=False)
pd.DataFrame(cen).to_csv("resultados/centroides.csv", index=False)
pd.DataFrame(asg).to_csv("resultados/asignaciones.csv", index=False)
print(M.round(3).to_string(index=False))

# Mejor k por silueta (entre 2 y 5) para cada par
mejor = M.loc[M.groupby("par").silueta.idxmax()][["par", "k", "silueta", "ARI_vs_especie"]]
print("\nMejor k por silueta:\n", mejor.round(3).to_string(index=False))

COL = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
# Gráficas de grupos: una figura por par, 4 paneles (k = 2..5)
for a, b in pares:
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    for p, k in zip(ax, KS):
        km, sc = modelos[(a, b, k)]
        for g in range(k):
            m = km.labels_ == g
            p.scatter(df[a][m], df[b][m], s=20, color=COL[g], alpha=.8, edgecolor="white", linewidth=.3)
        c = sc.inverse_transform(km.cluster_centers_)
        p.scatter(c[:, 0], c[:, 1], marker="X", s=150, color="k", edgecolor="white", zorder=5)
        r = M[(M.par == f"{a} / {b}") & (M.k == k)].iloc[0]
        p.set_title(f"k = {k}  |  silueta {r.silueta:.3f}", fontsize=10)
        p.set_xlabel(ES[a], fontsize=8); p.set_ylabel(ES[b], fontsize=8)
    fig.tight_layout(); fig.savefig(f"figs/grupos_{a}_{b}.png", dpi=150); plt.close(fig)

# Codo y silueta (6 curvas)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for par, g in M.groupby("par", sort=False):
    ax[0].plot(g.k, g.inercia, marker="o", label=par); ax[1].plot(g.k, g.silueta, marker="o", label=par)
ax[0].set_title("Método del codo (inercia)"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inercia")
ax[1].set_title("Coeficiente de silueta"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silueta")
for p in ax: p.set_xticks(KS); p.grid(alpha=.3)
ax[1].legend(fontsize=7); fig.tight_layout(); fig.savefig("figs/codo_silueta.png", dpi=150); plt.close(fig)

# Matriz comparativa (pares x k) de silueta
piv = M.pivot(index="par", columns="k", values="silueta").loc[M.par.unique()]
fig, a = plt.subplots(figsize=(6, 3.6))
im = a.imshow(piv.values, cmap="YlGnBu"); a.set_xticks(range(len(KS)), [f"k={k}" for k in KS])
a.set_yticks(range(len(piv)), piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]): a.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
fig.colorbar(im, label="Silueta"); fig.tight_layout(); fig.savefig("figs/matriz_silueta.png", dpi=150); plt.close(fig)
