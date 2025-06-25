# 📁 Data

This folder contains the datasets used in the project **Unheard Voices: Investigating Gender Bias in Music Recommendation Systems**. The data is derived from both **Spotify** and **Last.fm** platforms and includes gender-enriched metadata for fairness analysis.

---

## 📦 Spotify Datasets

- **`interactions.csv`**  
  Contains user-track interaction data extracted from the [Spotify Million Playlist Dataset](https://doi.org/10.1145/3269206.3269288).  
  Format: playlist-track pairs used for building the recommender system.

- **`playlist_id_map.csv`**  
  Maps playlist IDs to additional metadata, potentially including playlist names, creation date, or user IDs.

---

## 🎵 Last.fm Datasets

- **`Last.fm_data.csv`**  
  Raw user listening data from Last.fm. Includes artist-track-user interactions used for recommender modeling and fairness evaluation.

- **`artists.csv`**  
  Contains metadata about artists, such as names and IDs used for joining with gender information.

- **`lastfm_enriched_with_gender.csv`**  
  The Last.fm dataset enriched with gender metadata using the MusicBrainz API and custom mappings.  
  Used for evaluating demographic representation and fairness metrics.

- **`lfm-360-gender.json`**  
  A mapping file linking MusicBrainz artist IDs to gender labels (`Male`, `Female`, `Other`). Used to enrich the Last.fm dataset for bias analysis.

---

## 🔗 Data Sources

- **Spotify Million Playlist Dataset**  
  [https://doi.org/10.1145/3269206.3269288](https://doi.org/10.1145/3269206.3269288)

- **Last.fm**  
  [https://www.kaggle.com/datasets/harshal19t/lastfm-dataset](https://www.kaggle.com/datasets/harshal19t/lastfm-dataset)

- **MusicBrainz API**  
  Used to enrich artist metadata with gender attributes: [https://musicbrainz.org/](https://musicbrainz.org/)

---

## ⚠️ Notes

Some datasets are partially included due to size or licensing constraints. Refer to the project root `README.md` for full links and download instructions if needed.

