import pandas as pd

def cluster_analysis(data, vae_clusters, pca_clusters):
    df = pd.DataFrame({
        "song": data["song_names"],
        "language": data["language_labels"],
        "vae_cluster": vae_clusters,
        "pca_cluster": pca_clusters
    })
    return df
