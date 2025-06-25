# Unheard Voices 🎵
**Investigating Gender Bias in Music Recommendation Systems**

## 📌 Project Overview
This project investigates the presence and mitigation of **gender bias** in music recommender systems. We explore two real-world datasets (Spotify and Last.fm), assess fairness in popularity-based recommendation outputs, and apply algorithmic debiasing using **AIF360** and **Fairlearn**.

### 🎯 Research Questions
- Do music recommendation systems favor certain gender groups over others?
- How is artist gender distributed among the most frequently recommended tracks?
- Can algorithmic fairness techniques reduce bias in popularity-based recommenders?

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
- **Spotify**: [Million Playlist Dataset](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
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
