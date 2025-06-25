# 📁 Data

This folder contains the datasets used in the project **Unheard Voices: Investigating Gender Bias in Music Recommendation Systems**. The data is derived from both **Spotify** and **Last.fm** platforms and includes gender-enriched metadata for fairness analysis.

---

## 📦 Spotify Data

- **`interactions.csv`**  
  Contains user-track interaction data extracted from the [Spotify Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge) 
  Format: playlist-track pairs used for building the recommender system.

- **`playlist_id_map.csv`**  
  Maps playlist IDs to additional metadata, potentially including playlist names, creation date, or user IDs.

- **`spotify_tracks_with_gender.csv`** *(not included in repo, see Link)*  
  A large enriched dataset containing Spotify track metadata along with inferred or collected artist gender labels.  
  Used for fairness evaluation and gender-based analysis of recommendation exposure.  
  **Note**: Due to its large size, this file is not hosted in the GitHub repository.  
  👉 You can download it from: [Link](https://drive.google.com/file/d/1OZaKaUnG8-s9NwfGRnnc0ApUB14Rewah/view?usp=drive_link)
  
---

## 🎵 Last.fm Data

- **`Last.fm_data.csv`**  
  Raw user listening data from Last.fm. Includes artist-track-user interactions used for recommender modeling and fairness evaluation.

- **`artists.csv`**  *(not included in repo, see Link)*
  Contains metadata about artists, such as names and IDs used for joining with gender information.lij
  **Note**: Due to its large size, this file is not hosted in the GitHub repository.  
  👉 You can download it from: [Link](https://drive.google.com/file/d/1z8mnEmztfy_FqOv1R71fC5KXesjNmU_u/view?usp=drive_link)
  
- **`lastfm_enriched_with_gender.csv`**  
  The Last.fm dataset enriched with gender metadata using the MusicBrainz API and custom mappings.  
  Used for evaluating demographic representation and fairness metrics.

- **`lfm-360-gender.json`**  
  A mapping file linking MusicBrainz artist IDs to gender labels (`Male`, `Female`, `Other`). Used to enrich the Last.fm dataset for bias analysis.

---

## 🔗 Data Sources

- **Spotify Million Playlist Dataset**  
  [https://doi.org/10.1145/3269206.3269288](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)

- **Last.fm**  
  [https://www.kaggle.com/datasets/harshal19t/lastfm-dataset](https://www.kaggle.com/datasets/harshal19t/lastfm-dataset)

- **MusicBrainz API**  
  Used to enrich artist metadata with gender attributes: [https://musicbrainz.org/](https://musicbrainz.org/)

---

## ⚠️ Notes

Some datasets are partially included due to size or licensing constraints. Refer to the project root `README.md` for full links and download instructions if needed.

