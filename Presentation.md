# 🎤 Presentation Guide: Unheard Voices – Investigating Gender Bias in Music Recommendation Systems

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

| Metric                 | Meaning                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| **Exposure Ratio**     | Compares how often items from each gender appear in recommendations |
| **Disparate Impact**   | Ratio of positive outcomes across groups (ideal: close to 1)        |
| **Demographic Parity** | Checks if gender affects chance of being recommended                |
| **Equal Opportunity**  | Evaluates fairness conditional on artist quality or relevance       |


---

# 📊 Project Summary & Results Comparison

## Overview of Files
| Filename                                                                            | Dataset | Purpose / Focus                                                 | Recommender Type | Fairness Tool |
| ----------------------------------------------------------------------------------- | ------- | --------------------------------------------------------------- | ---------------- | ------------- |
| `01_Spotify_GenderBiasinMusicRecommenderSystems_Data Retrieval_Preprocessing.ipynb` | Spotify | Data retrieval, preprocessing, gender enrichment via API        | –                | –             |
| `02_Spotify_descriptive_data_incl_files.ipynb`                                      | Spotify | Exploratory Data Analysis (EDA) and descriptive statistics      | –                | –             |
| `03_Spotify_Baselinerecommender_Fairlearn.ipynb`                                    | Spotify | Popularity-based recommender + fairness evaluation              | Baseline         | Fairlearn     |
| `04_Spotify_ContentBased Filtering_recommender_Fairlearn.ipynb`                     | Spotify | Content-based filtering + post-processing fairness improvements | Content-Based    | Fairlearn     |
| `05.1_Spotify_recommender_AIF.ipynb`                                                | Spotify | Popularity-based fairness analysis using AIF360                 | Baseline         | AIF360        |
| `05.2_Spotify_ContentBased Filtering_recommender_AIF360.ipynb`                      | Spotify | Content-based filtering fairness mitigation via AIF360          | Content-Based    | AIF360        |
| `01_LastFm_DataRetrieval_Preprocessing.ipynb`                                       | Last.fm | Data retrieval, cleaning, gender mapping                        | –                | –             |
| `02_Last.fm_Baselinerecommender_Fairlearn.ipynb`                                    | Last.fm | Baseline recommender + fairness evaluation                      | Baseline         | Fairlearn     |
| `03_Last.fm_ContentBasedFiltering_Fairlearn.ipynb`                                  | Last.fm | Content-based recommender + Fairlearn-based bias mitigation     | Content-Based    | Fairlearn     |
| `04_Last.fm_AIF360.ipynb`                                                           | Last.fm | Content-based recommender with AIF360 fairness pipeline         |baseline & Content-Based    | AIF360        |



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
| `04_Last.fm_AIF360.ipynb`                           | Popularity & Content-based    | AIF360    | ✅ Yes     | ✅ Significantly | Most effective for Last.fm data |

---

## 🧾 ** Summary of most important Project Findings**

### 1. 🎯 **Gender Bias Is Real and Measurable**

* Both **Spotify** and **Last.fm** recommendation systems displayed a **clear gender bias**, favoring **male artists**.
* **Female and non-binary artists** were significantly underrepresented, especially in **baseline (popularity-based)** models.

> Example: Female artist exposure often fell below **40%** in top-ranked recommendations.

---

### 2. 📊 **Fairness Metrics Quantify Bias Clearly**

* **Exposure Ratio** and **Disparate Impact** were the most informative for the domain.
* **Demographic Parity** and **Equal Opportunity** violations helped pinpoint where and how the system failed to treat gender groups equally.

> In several cases, **disparate impact was below 0.8**, indicating a legally and ethically significant imbalance.

---

### 3. 🤖 **Fairness Tools Can Mitigate Bias**

| Tool           | Result                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------ |
| **Fairlearn**  | Improved fairness with **re-ranking**, especially in content-based models. Some trade-off in accuracy.       |
| **AIF360**     | Achieved **strong fairness gains** in both baseline and content-based models with **minimal accuracy loss**. |
| **Best Setup** | **Content-Based Filtering + AIF360** gave the **most balanced** outcome between fairness and performance.    |

---

### 4. 📁 **Cross-Dataset Findings Are Consistent**

* **Bias was present in both datasets**, though Spotify’s baseline bias was slightly more pronounced.
* **Fairness tools generalized well** across datasets, with AIF360 consistently outperforming in mitigation strength.

---

### 5. ✅ **Fairness ≠ Accuracy Trade-Off Is Manageable**

* Applying fairness-aware techniques **did not drastically reduce recommender performance**.

---

## 🧠 Conclusion

- **Bias is clearly present** in both datasets and recommender types.
- **AIF360 outperformed Fairlearn** in both popularity-based and content-based recommenders.
- **Content-based filtering + AIF360** yielded the **best fairness-to-accuracy balance**.
- **Spotify and Last.fm** both benefited from fairness-aware re-ranking, but improvements varied by dataset and tool.


- **Fairness vs. Accuracy**: Minimal drop in performance when fairness tools were applied.
- **Most Effective**: AIF360 showed **strong fairness improvements** with **minimal accuracy trade-off**.

---

## ✅ Research Question Summary & Findings

### 🎯 **Main Research Question:**

**Does gender bias exist in music recommender systems?**
➡️ **Yes.**
Both the Spotify and Last.fm datasets revealed a **systematic underrepresentation** of female and non-binary artists in recommendation results, particularly in baseline (popularity-based) models. Metrics like exposure ratio and disparate impact consistently showed that **male artists were disproportionately favored**.

---

### 🔍 **Sub-Research Question 1:**

**How do different fairness metrics (e.g., demographic parity, equal opportunity) reflect imbalances?**

➡️ **Each metric highlights different aspects of bias**:

| Metric                 | What It Shows                                                                 | Example Finding                             |
| ---------------------- | ----------------------------------------------------------------------------- | ------------------------------------------- |
| **Exposure Ratio**     | Whether different genders are shown to users at comparable rates              | Female artists had <40% exposure on average |
| **Demographic Parity** | Measures whether recommendations are independent of gender                    | Violated in most baseline systems           |
| **Equal Opportunity**  | Evaluates if the recommender equally promotes qualified items from all groups | Male artists more likely to be top-ranked   |
| **Disparate Impact**   | Ratio of favorable outcomes between groups                                    | Often <0.8 (threshold for fairness)         |

These metrics provide **quantitative evidence** of structural imbalance, allowing for **targeted mitigation**.

---

### 🤖 **Sub-Research Question 2:**

**How do different fairness-aware tools perform in mitigating gender bias in music recommendation systems?**

➡️ **Performance Comparison**:

| Tool                  | Recommender Type           | Performance Summary                                                                       |
| --------------------- | -------------------------- | ----------------------------------------------------------------------------------------- |
| **Fairlearn**         | Post-processing re-ranking | Improved exposure for underrepresented artists, but often with minor accuracy loss        |
| **AIF360**            | Bias mitigation toolkit    | Provided **stronger fairness improvements** across most configurations                    |
| **Combined Strategy** | Fairlearn + AIF360         | When using both methods in succession this can yielded the **most balanced results** |

* **Fairlearn** was more transparent and modular (e.g., useful for re-ranking).
* **AIF360** included broader fairness checks (e.g., equal opportunity, statistical parity), and produced **the most significant bias reduction** with minimal accuracy loss in content-based recommenders.

---

## 🧠 Final Insight

✅ Gender bias is **measurable and real** in music recommendation systems.
⚙️ **Fairness-aware tools work**, and **choice of metric/tool matters** depending on the fairness objective.
📈 The **most effective approach** combined content-based filtering with **AIF360**, achieving fairness **without sacrificing recommendation quality**.


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
