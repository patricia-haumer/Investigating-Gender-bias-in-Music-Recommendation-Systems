# 🎤 Presentation Guide: Unheard Voices – Gender Bias in Music Recommendation Systems

## 🎯 1. Project Motivation

- **Problem:** Music recommender systems often favor male artists.
- **Evidence:** Users wait until track 7–8 to hear a female artist (Ferraro et al., 2021).
- **Impact:** Reinforces inequality, reduces visibility for underrepresented artists.

✅ **Goal:** Detect and mitigate gender bias to create fairer recommendation systems.

---

## 🔍 2. Research Questions

1. Does **gender bias** exist in music recommender systems?
2. How do **fairness metrics** like demographic parity reflect this bias?
3. Can tools like **Fairlearn** and **AIF360** reduce bias without hurting performance?

---

## 🧪 3. Methodology

- **Framework:** Design Science Research (Peffers et al., 2007)
- **Tools:** Python, Jupyter, Fairlearn, AIF360
- **Datasets:** Spotify Million Playlist Dataset + Last.fm (with gender enrichment)

---

## 🛠️ 4. Recommender Systems Tested

### 📈 Spotify:
- ✅ Baseline: Popularity-based (Fairlearn)
- ✅ Content-Based: Artist & Genre similarity (Fairlearn, AIF360)

### 🎵 Last.fm:
- ✅ Baseline: Popularity (Fairlearn)
- ✅ Content-Based: Similarity-based (Fairlearn, AIF360)

---

## 📊 5. Evaluation Metrics

- **Fairness:**
  - Demographic Parity
  - Exposure Ratio
  - Disparate Impact
- **Performance:**
  - Accuracy
  - Recommendation Coverage

---

## 🧠 6. Key Findings

| Dataset | Method                  | Fairness Improved? | Accuracy Impact |
|---------|--------------------------|---------------------|------------------|
| Spotify | Fairlearn + CBF          | ✅ Yes               | ⬇️ Minimal       |
| Spotify | AIF360 + CBF             | ✅ Yes               | ⬇️ Minimal       |
| Last.fm | Fairlearn Baseline       | ✅ Yes               | ⬇️ Minimal       |
| Last.fm | AIF360 Content-Based     | ✅ Yes               | ⬇️ Minimal       |

✅ Bias mitigation is effective **without major loss in performance**.

---

## 🚀 7. Contribution

- Highlighted **real gender bias** in popular music platforms
- Applied **multiple fairness frameworks** for mitigation
- Created a **reproducible workflow** for bias analysis in recommendations

---

## 📌 8. Future Work

- Try **RecBole** or **FaiRecSys** for deeper recommender integration
- Expand analysis to **non-binary** and intersectional identities
- Test on **real user interaction logs**

---

## 🙌 9. Team & Acknowledgements

👩‍💻 Katharina Rosa Pöcher  
👩‍💻 Patricia Haumer  

Special thanks to WU Vienna and the Data Science & AI II course.

---

## 🔗 10. GitHub Repository

➡️ [GitHub Project Link](https://github.com/YOUR_USERNAME/unheard-voices-recommender)

