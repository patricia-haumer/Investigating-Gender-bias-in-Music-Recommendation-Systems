# 📊 Project Summary & Results Comparison

## 🎧 Spotify Dataset

| Notebook                                                   | Recommender Type | Tool Used | Bias Found | Bias Mitigated | Notes |
|------------------------------------------------------------|------------------|-----------|------------|----------------|-------|
| `03_Spotify_Baselinerecommender_Fairlearn.ipynb`           | Popularity-based | Fairlearn | ✅ Yes     | ⚠️ Partially    | Exposure gap present between male/female artists |
| `04_Spotify_ContentBased Filtering_recommender_Fairlearn.ipynb` | Content-based    | Fairlearn | ✅ Yes     | ✅ Significantly | Improved exposure ratio |
| `05.1_Spotify_recommender_AIF.ipynb`                       | Popularity-based | AIF360    | ✅ Yes     | ✅ Yes         | Better fairness metrics vs. Fairlearn |
| `05.2_Spotify_ContentBased Filtering_recommender_AIF360.ipynb` | Content-based    | AIF360    | ✅ Yes     | ✅ Strongly     | Most balanced fairness/accuracy trade-off |

## 🎵 Last.fm Dataset

| Notebook                                            | Recommender Type | Tool Used | Bias Found | Bias Mitigated | Notes |
|-----------------------------------------------------|------------------|-----------|------------|----------------|-------|
| `02_Last.fm_Baselinerecommender_Fairlearn.ipynb`    | Popularity-based | Fairlearn | ✅ Yes     | ⚠️ Partially    | Gender bias visible, not fully corrected |
| `03_Last.fm_ContentBasedFiltering_Fairlearn.ipynb`  | Content-based    | Fairlearn | ✅ Yes     | ✅ Yes         | Notable fairness gain via re-ranking |
| `04_Last.fm_AIF360.ipynb`                           | Content-based    | AIF360    | ✅ Yes     | ✅ Significantly | Most effective for Last.fm data |

---

## 🧠 Conclusion

- **Bias is clearly present** in both datasets and recommender types.
- **AIF360 outperformed Fairlearn** in both popularity-based and content-based recommenders.
- **Content-based filtering + AIF360** yielded the **best fairness-to-accuracy balance**.
- **Spotify and Last.fm** both benefited from fairness-aware re-ranking, but improvements varied by dataset and tool.

