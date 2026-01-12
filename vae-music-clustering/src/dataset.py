import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class HybridMusicDataset:
    """Create hybrid English+Bangla music dataset with audio and lyrics features"""

    def __init__(self, n_samples=200):
        self.n_samples = n_samples
        self.n_english = n_samples // 2
        self.n_bangla = n_samples // 2

    def _create_english_samples(self):
        samples = []
        for i in range(self.n_english):
            audio = np.random.randn(260) * 0.5
            if i % 2 == 0:
                audio[:65] += np.random.randn(65) * 0.8
                audio[195:] -= np.random.randn(65) * 0.4
            else:
                audio[65:130] += np.random.randn(65) * 0.9
            samples.append(audio)
        return np.array(samples)

    def _create_bangla_samples(self):
        samples = []
        for i in range(self.n_bangla):
            audio = np.random.randn(260) * 0.6
            if i % 2 == 0:
                audio[:100] += np.random.randn(100) * 0.7
                audio[150:200] += np.random.randn(50) * 0.5
            else:
                audio[50:150] += np.random.randn(100) * 0.8
                audio[-100:] -= np.random.randn(100) * 0.3
            samples.append(audio)
        return np.array(samples)

    def _create_lyrics_features(self):
        english_lyrics = [
            "love heart beat night dream",
            "dance party fun time friends",
            "rain tears pain sad alone",
            "sky fly high freedom wind",
            "city lights night life energy",
            "ocean waves beach summer sun",
            "mountain climb adventure journey",
            "star moon night romance kiss",
            "road travel journey freedom",
            "fire burn passion desire flame"
        ] * (self.n_english // 10)

        bangla_lyrics = [
            "ami tomay bhalobashi mon pran",
            "bangla desh matribhumi sonar",
            "megher kole rod shekhe bhalobasa",
            "je rate mor duaruli gaan shuni",
            "phiriye dao amay chhere dao bondhu",
            "jodi tor dak shune keu na ashe",
            "ami chini go chini tomare ogo",
            "o pori hobe na din katena keno",
            "mon majhe rekhe geli bone bashonti",
            "shei to ami noi alo andhare prithibi"
        ] * (self.n_bangla // 10)

        all_lyrics = english_lyrics[:self.n_english] + bangla_lyrics[:self.n_bangla]
        lyrics_labels = [0]*self.n_english + [1]*self.n_bangla

        vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        lyrics_features = vectorizer.fit_transform(all_lyrics).toarray()

        return lyrics_features, lyrics_labels, all_lyrics

    def create_dataset(self):
        english_audio = self._create_english_samples()
        bangla_audio = self._create_bangla_samples()
        all_audio = np.vstack([english_audio, bangla_audio])

        lyrics_features, lyrics_labels, lyrics_text = self._create_lyrics_features()

        true_labels = (
            [0 if i % 2 == 0 else 1 for i in range(self.n_english)] +
            [2 if i % 2 == 0 else 3 for i in range(self.n_bangla)]
        )

        language_labels = [0]*self.n_english + [1]*self.n_bangla
        combined_features = np.hstack([all_audio, lyrics_features])

        song_names = (
            [f"English_{'Pop' if i%2==0 else 'Rock'}_{i}" for i in range(self.n_english)] +
            [f"Bangla_{'Rabindra' if i%2==0 else 'Modern'}_{i}" for i in range(self.n_bangla)]
        )

        return {
            "features": combined_features.astype(np.float32),
            "true_labels": np.array(true_labels),
            "language_labels": np.array(language_labels),
            "song_names": song_names,
            "lyrics_text": lyrics_text
        }
