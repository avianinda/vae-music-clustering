import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def visualize(latent, clusters, labels):
    tsne = TSNE(n_components=2, random_state=42)
    umap_embed = umap.UMAP(n_components=2, random_state=42)

    tsne_z = tsne.fit_transform(latent)
    umap_z = umap_embed.fit_transform(latent)

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.scatter(tsne_z[:,0], tsne_z[:,1], c=labels, cmap="coolwarm")
    plt.title("t-SNE (Language)")

    plt.subplot(1,2,2)
    plt.scatter(umap_z[:,0], umap_z[:,1], c=clusters, cmap="tab10")
    plt.title("UMAP (Clusters)")
    plt.show()

    return tsne_z, umap_z
