# Unheard Voices 🎵
**Investigating Gender Bias in Music Recommendation Systems**

## 📌 Project Overview
This project investigates the presence and mitigation of **gender bias** in music recommender systems. We explore two real-world datasets (Spotify Million Playlist Dataset and Last.fm), assess fairness in popularity-based recommendation outputs, and apply algorithmic debiasing using **AIF360** and **Fairlearn**.

### 🎯 Project Goals
- Analyze the **gender distribution** in Spotify and Last.fm music data.
- Build **baseline recommenders** based on item popularity.
- Quantitatively evaluate fairness using **statistical metrics** (e.g., disparate impact).
- Apply **bias mitigation strategies** (e.g., Reweighing, Fairlearn reductions).
- Promote **fairer exposure** for underrepresented gender groups in algorithmic music curation.

### ❓ Research Questions
- Does gender bias exist in music recommender systems?

**Sub-Research Questions:**
- How do different fairness metrics e.g. demographic parity, equal opportunity reflect imbalances?​
- How do different fairness-aware tools perform in mitigating gender bias in music recommendation systems

## 🧪 Methodology

We followed the **Design Science Research** approach (Peffers et al., 2007):

1. **Problem Identification** – Gender bias in music recommender systems.
2. **Design & Development** – Baseline + Content-Based recommenders using Spotify & Last.fm.
3. **Evaluation** – Fairness and accuracy metrics (exposure ratio, disparate impact, etc.).
4. **Tools Used**: `Fairlearn`, `AIF360`, `FaiRecSys`, `RecBole`.

## 📁 Structure
- `data/`: contains the datasets used or links to download them
- `notebooks/`: contains data processing, descriptive statistics, modeling, and fairness evaluation notebooks
- `requirements`: packages needed to run the notebooks
- `License`: open-source license for usage and citation
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

## 📈 Results Summary

✅ Gender bias **detected** in both baseline and content-based recommenders.
⚙️ Fairness-aware re-ranking methods **reduced bias** while preserving recommendation quality.
🎯 AIF360 and Fairlearn metrics guided bias mitigation strategies effectively.

## 👩‍💻 Contributors
- Patricia Haumer (h11910653)
- Katharina Rosa Pöcher (h11917060)
