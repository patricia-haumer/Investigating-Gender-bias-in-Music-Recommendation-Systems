# Unheard Voices 🎵
**Investigating Gender Bias in Music Recommendation Systems**

## 📌 Project Overview
This project investigates the presence and mitigation of **gender bias** in music recommender systems. We explore two real-world datasets (Spotify and Last.fm), assess fairness in popularity-based recommendation outputs, and apply algorithmic debiasing using **AIF360** and **Fairlearn**.

### 🎯 Research Questions

**Main Research Question:**
- Does gender bias exist in music recommender systems?

**Sub-Research Questions:**
- How is artist gender distributed among the most frequently recommended tracks?
- How do different fairness metrics (e.g., demographic parity, equal opportunity) reflect imbalances?
- How do different fairness-aware tools (e.g., AIF360, Fairlearn) perform in mitigating gender bias in music recommendation systems?
- Can algorithmic fairness techniques improve the exposure of underrepresented gender groups without sacrificing recommendation quality?


### 🎯 Project Goals
- Analyze the **gender distribution** in Spotify and Last.fm music data.
- Build **baseline recommenders** based on item popularity.
- Quantitatively evaluate fairness using **statistical metrics** (e.g., disparate impact).
- Apply **bias mitigation strategies** (e.g., Reweighing, Fairlearn reductions).
- Promote **fairer exposure** for underrepresented gender groups in algorithmic music curation.

## 📁 Structure
- `data/`: contains the datasets used or links to download them
- `notebooks/`: contains data processing, descriptive statistics, modeling, and fairness evaluation notebooks
- `requirements.txt`: packages needed to run the notebooks
- `README.md`: this file

## 📊 Datasets
- **Spotify**: A dataset of 1 million user-generated playlists (2010–2017) used to analyze track popularity and gender representation in music recommendations [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
- **Last.fm**: Collected via publicly available sources from [Kaggle](https://www.kaggle.com/datasets/harshal19t/lastfm-dataset)
- The datasets were enriched using the [MusicBrainz API](https://musicbrainz.org/), by retrieving the gender attributes

Some data files (due to size or licensing) may be partially included or require external download.

## 🧠 Tools & Libraries
- Python (pandas, numpy, scikit-learn)
- [Fairlearn](https://github.com/fairlearn/fairlearn) 
- [AIF360](https://github.com/Trusted-AI/AIF360)
- Jupyter Notebooks

## 👩‍💻 Contributors
- Katharina Rosa Pöcher (h11917060)
- Patricia Haumer (h11910653)
