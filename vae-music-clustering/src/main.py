from dataset import HybridMusicDataset
from train import train_vae
from clustering import evaluate_clustering
from visualization import visualize

dataset = HybridMusicDataset()
data = dataset.create_dataset()

model, latent, scaler = train_vae(data)

results = evaluate_clustering(
    latent,
    data["features"],
    data["language_labels"],
    scaler
)

visualize(latent, results["vae_clusters"], data["language_labels"])
