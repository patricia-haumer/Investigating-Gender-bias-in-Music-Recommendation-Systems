# 📓 Project Notebooks

This directory contains all Jupyter notebooks used in the project, structured by dataset and analysis stage.

---

## 🎧 Spotify Notebooks

| Filename                                                              | Description                                                                 |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_Spotify_GenderBiasinMusicRecommenderSystems_Data Retrieval_Preprocessing.ipynb` | Data loading, cleaning, and gender enrichment using MusicBrainz API        |
| `02_Spotify_descriptive_data_incl_files.ipynb`                        | Descriptive analysis on gender distribution, playlist features, and track ranking |
| `03_Spotify_Baselinerecommender_Fairlearn.ipynb`                      | Popularity-based recommender + fairness analysis with Fairlearn             |
| `04_Spotify_ContentBased Filtering_recommender_Fairlearn.ipynb`       | Content-based filtering + fairness enhancement using Fairlearn              |
| `05.1_Spotify_recommender_AIF.ipynb`                                  | Fairness analysis of Spotify baseline recommender using AIF360              |
| `05.2_Spotify_ContentBased Filtering_recommender_AIF360.ipynb`        | Content-based recommender with bias mitigation via AIF360                   |

---

## 🎵 Last.fm Notebooks

| Filename                                             | Description                                                                 |
|------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_LastFm_DataRetrieval_Preprocessing.ipynb`        | Last.fm data extraction, cleaning, gender augmentation                      |
| `02_Last.fm_Baselinerecommender_Fairlearn.ipynb`     | Popularity-based recommender + Fairlearn fairness evaluation                |
| `03_Last.fm_ContentBasedFiltering_Fairlearn.ipynb`   | Content-based filtering recommender + Fairlearn re-ranking                  |
| `04_Last.fm_AIF360.ipynb`                            | AIF360 fairness mitigation for Last.fm recommender system                   |

---

## 🛠️ Notes

- All notebooks are self-contained and reproducible.
- Place required datasets in the `data/` folder.
- Use `pip install -r requirements.txt` before running the notebooks.
