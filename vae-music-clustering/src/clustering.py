import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, confusion_matrix

def evaluate_clustering(latent, original_features, labels, scaler, n_clusters=4):
    vae_clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(latent)

    X_scaled = scaler.transform(original_features)
    pca_features = PCA(n_components=latent.shape[1]).fit_transform(X_scaled)
    pca_clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(pca_features)

    def metrics(features, clusters, language):
        sil = silhouette_score(features, clusters)
        cal = calinski_harabasz_score(features, clusters)
        cm = confusion_matrix(language, clusters)
        purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)
        return sil, cal, purity

    return {
        "vae_clusters": vae_clusters,
        "pca_clusters": pca_clusters,
        "metrics": {
            "vae": metrics(latent, vae_clusters, labels),
            "pca": metrics(pca_features, pca_clusters, labels)
        }
    }
