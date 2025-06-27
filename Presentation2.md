# 🎤 Presentation Guide: Unheard Voices – Investigating Gender Bias in Music Recommendation Systems

This document outlines key talking points and content for your 10-minute presentation.

---

## 🎯 Project Motivation

- **Problem**: Algorithmic music recommenders often **amplify existing gender biases**, leading to reduced exposure for **female and non-binary artists**.
- **Example**: “On average, users had to wait until song 7 or 8 to hear one by a woman.” *(Ferraro et al., 2021)*
- **Societal Relevance**: Aligns with EU’s **AI Act**, requiring bias detection and mitigation in AI systems.
- **Business Case**: Fairer recommendations can enhance user trust, engagement, and platform reputation.

---

## ❓ Research Questions

1. **Does gender bias exist** in music recommender systems?
2. How do **fairness metrics** (e.g., exposure ratio, demographic parity) reveal imbalances?
3. How do various **fairness-aware tools** (Fairlearn, AIF360, etc.) perform in reducing bias?

---

## 🧪 Methodology

- Based on **Design Science Research** (Peffers et al., 2007)
- **Steps**:
  - Problem identification and goal setting
  - Recommender system design (baseline and content-based)
  - Application of fairness-aware methods (Fairlearn, AIF360)
  - Evaluation using both **accuracy and fairness metrics**
  - Validation on two datasets: **Spotify** and **Last.fm**

---

## 📂 Datasets

- **Spotify Million Playlist Dataset**: Enriched with artist gender via **MusicBrainz API**
- **Last.fm Dataset**: User listening history; used for validation
- Gender metadata: Categorized into *male*, *female*, and *non-binary*; encoded for analysis

---

## ⚙️ Tools & Techniques

- **Recommender Types**:
  - Baseline (Popularity-based)
  - Content-Based Filtering
- **Fairness Tools**:
  - `Fairlearn` – post-processing re-ranking
  - `AIF360` – bias detection and mitigation
- **Evaluation Metrics**:
  - Accuracy: Precision, Recall
  - Fairness: Exposure Ratio, Disparate Impact, Demographic Parity

---

## 📊 Key Findings

| Dataset | Recommender Type | Tool Used  | Bias Found? | Bias Mitigated? |
|---------|------------------|------------|-------------|-----------------|
| Spotify | Baseline         | Fairlearn  | ✅ Yes      | ✅ Partially     |
| Spotify | Content-Based    | AIF360     | ✅ Yes      | ✅ Significantly |
| Last.fm | Baseline         | Fairlearn  | ✅ Yes      | ✅ Partially     |
| Last.fm | Content-Based    | AIF360     | ✅ Yes      | ✅ Significantly |

- **Fairness vs. Accuracy**: Minimal drop in performance when fairness tools were applied.
- **Most Effective**: AIF360 showed **strong fairness improvements** with **minimal accuracy trade-off**.

---

## 🧠 Conclusion

- Gender bias exists and persists in music recommendation systems.
- Fairness-aware algorithms **can mitigate bias** without severely harming recommendation quality.
- Project supports the development of **more equitable AI systems**, aligned with **societal and legal expectations**.

---

## 🧩 Limitations & Outlook

- Limited gender label availability (resolved via MusicBrainz API)
- Only binary gender used in current implementation
- Future work: Deep learning-based recommenders, intersectional fairness (e.g., gender + genre)

---

## 🗂️ Resources

- [Full Project on GitHub](https://github.com/YOUR_USERNAME/unheard-voices-recommender)
- Tools: `Fairlearn`, `AIF360`, `Pandas`, `Scikit-learn`, `RecBole`, `MusicBrainz API`

---

## 👩‍💻 Authors

- **Katharina Rosa Pöcher** – h11917060
- **Patricia Haumer** – h11910653

Presented as part of the course **Data Science & AI II** at **WU Vienna**.
