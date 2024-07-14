import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score
from sklearn.datasets import make_blobs

# Création de données simulées
n_samples = 300
n_features = 5
n_clusters = 4

X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Taille des données
print("Taille des données : ", X.shape)

# Initialisation Aléatoire
kmeans_random = KMeans(n_clusters=n_clusters, init='random', random_state=42)
kmeans_random.fit(X)

# Initialisation K-means++
kmeans_plus = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans_plus.fit(X)

# Méthodes de Validation de Clustering
score_random = calinski_harabasz_score(X, kmeans_random.labels_)
score_plus = calinski_harabasz_score(X, kmeans_plus.labels_)

print("Score Calinski-Harabasz (Initialisation Aléatoire) : ", score_random)
print("Score Calinski-Harabasz (K-means++) : ", score_plus)

# Meilleur Modèle de Clustering
best_model = kmeans_plus if score_plus > score_random else kmeans_random

# Analyse en Composantes Principales (ACP)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
centers_pca = pca.transform(best_model.cluster_centers_)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_model.labels_, s=50, cmap='viridis')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.75)
plt.title('Projection des données et des centres avec ACP')
plt.xlabel('Premier Axe Principal')
plt.ylabel('Deuxième Axe Principal')
plt.show()

# Calcul des Valeurs Propres et Vecteurs Propres
print("Valeurs propres : ", pca.explained_variance_)
print("Vecteurs propres : ", pca.components_)

# Inertie de Chaque Axe et Vérification
inertie = np.sum(pca.explained_variance_)
print("Inertie de chaque axe : ", pca.explained_variance_)
print("Somme des inerties : ", inertie)
print("Dimension des données : ", n_features)