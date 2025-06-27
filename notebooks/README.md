# 📓 Project Notebooks

This directory contains all Jupyter notebooks used in the project, structured by dataset and analysis stage.

## 🎧 Spotify Notebooks

| Filename                                                     | Description                                                                 |
|--------------------------------------------------------------|-----------------------------------------------------------------------------|
| `01_Spotify_GenderBiasinMusicRecommenderSystems_Data Retrieval_Preprocessing.ipynb` | Spotify data loading, cleaning, and gender enrichment                      |
| `02_Spotify_descriptive_data_incl_files.ipynb`               | Descriptive analysis on gender, playlists, and track positioning            |
| `03_Spotify_Baselinerecommender_Fairlearn.ipynb`             | Popularity-based baseline recommender system + fairness analysis with Fairlearn |
| `04_Spotify_ContentBased Filtering_recommender_Fairlearn.ipynb` | Content-based filtering recommender + fairness mitigation with Fairlearn |
| `05_Spotify_ContentBased Filtering_recommender_AIF360.ipynb` | Content-based filtering recommender + fairness mitigation with AIF360       |

## 🎵 Last.fm Notebooks

| Filename                                       | Description                                                                 |
|------------------------------------------------|-----------------------------------------------------------------------------|
| `01_LastFm_DataRetrieval_Preprocessing.ipynb` | Last.fm data retrieval, cleaning, and gender enrichment                      |
| `02_Last.fm_Baselinerecommender_Fairlearn.ipynb` | Popularity-based baseline recommender + Fairlearn fairness analysis         |
| `03_Last.fm_ContentBasedFiltering_Fairlearn.ipynb` | Content-based filtering with fairness evaluation via Fairlearn             |
| `04_Last.fm_AIF360.ipynb`                     | Bias detection and mitigation using AIF360 on Last.fm content               |

## 🛠️ Notes

- All notebooks are self-contained and reproducible.
- Data should be placed in the `data/` folder as described in the root `README.md`.
- Run `pip install -r requirements.txt` before using the notebooks.
