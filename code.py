import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# RFM table built from the UCI Online Retail dataset (one row per customer)
df = pd.read_csv("dataset.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Frequency and Monetary are very skewed, so take log before scaling
X = np.log1p(df[["Recency", "Frequency", "Monetary"]])
X = StandardScaler().fit_transform(X)

# choose k using elbow + silhouette
inertia = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
    print("k =", k, "silhouette =", round(silhouette_score(X, km.labels_), 3))

plt.plot(range(2, 9), inertia, marker="o")
plt.xlabel("k")
plt.ylabel("inertia")
plt.title("Elbow method")
plt.savefig("elbow.png")
plt.close()

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
df["KMeans"] = kmeans.fit_predict(X)

dbscan = DBSCAN(eps=0.5, min_samples=10)
df["DBSCAN"] = dbscan.fit_predict(X)
print("DBSCAN clusters:", df["DBSCAN"].nunique() - 1, " noise points:", (df["DBSCAN"] == -1).sum())
mask = df["DBSCAN"] != -1
print("KMeans silhouette:", round(silhouette_score(X, df["KMeans"]), 3))
print("DBSCAN silhouette (without noise):", round(silhouette_score(X[mask], df["DBSCAN"][mask]), 3))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("PCA explained variance:", pca.explained_variance_ratio_, "total:", pca.explained_variance_ratio_.sum())

X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].scatter(X_pca[:, 0], X_pca[:, 1], c=df["KMeans"], s=5, cmap="tab10")
ax[0].set_title("K-Means on PCA")
ax[1].scatter(X_pca[:, 0], X_pca[:, 1], c=df["DBSCAN"], s=5, cmap="tab10")
ax[1].set_title("DBSCAN on PCA (-1 = noise)")
ax[2].scatter(X_tsne[:, 0], X_tsne[:, 1], c=df["KMeans"], s=5, cmap="tab10")
ax[2].set_title("K-Means on t-SNE")
plt.savefig("clusters.png")
plt.close()

summary = df.groupby("KMeans")[["Recency", "Frequency", "Monetary"]].mean().round(1)
summary["Customers"] = df["KMeans"].value_counts()
print(summary)

# personas: rank clusters by average spend (highest spend first)
personas = ["Champions - recent, frequent, big spenders -> loyalty rewards, early access, no discounts needed",
            "Potential loyalists - regular buyers, cooling down -> personalised recommendations, bundles",
            "New customers - bought recently but once or twice -> welcome offer to get the 2nd order",
            "Lost / hibernating - long time no purchase, low spend -> cheap win-back email only"]
order = summary.sort_values("Monetary", ascending=False).index
for c, p in zip(order, personas):
    print("Cluster", c, ":", p)

print("K-Means forces every customer into a cluster and gives clear segments for marketing.")
print("DBSCAN finds dense groups and marks unusual customers (very big spenders) as noise,")
print("which is useful for spotting outliers but the segments are less balanced.")
print("PCA is used for a quick linear 2D view, t-SNE keeps local neighbours so groups separate better visually.")
