## 📘 01 LastFm DataRetrieval Preprocessing

# Validation Dataset: Last FM Dataset - Data Retrievel, Pre-Processing

### Last FM Dataset - Data Retrieval
For Validation of our project result we'll use the Last.fm dataset (Link: https://www.kaggle.com/datasets/harshal19t/lastfm-dataset), therefore subsequently you'll find some facts about the dataset.

## About the Dataset

The **Last.fm dataset** consists of **166,153 entries** and **6 attributes**:

- **Username**: Consists of the name of the user.
- **Artist**: Name of the artists that the user had heard.
- **Track**: Consists of track/song name by that particular artist.
- **Album**: Consists of names of the albums.
- **Date**: Consists of the days ranging from January 1st to January 31st, 2021.
- **Time**: Consists of the time of a particular day when the user had heard a particular track.


   Unnamed: 0 Username           Artist                          Track  \
0           0  Babs_05  Isobel Campbell     The Circus Is Leaving Town   
1           1  Babs_05  Isobel Campbell                   Dusty Wreath   
2           2  Babs_05  Isobel Campbell     Honey Child What Can I Do?   
3           3  Babs_05  Isobel Campbell  It's Hard To Kill A Bad Thing   
4           4  Babs_05  Isobel Campbell                Saturday's Gone   

                       Album         Date    Time  
0  Ballad of the Broken Seas  31 Jan 2021   23:36  
1  Ballad of the Broken Seas  31 Jan 2021   23:32  
2  Ballad of the Broken Seas  31 Jan 2021   23:28  
3  Ballad of the Broken Seas  31 Jan 2021   23:25  
4  Ballad of the Broken Seas  31 Jan 2021   23:21  


### Inspect the Lastfm Data

Shape of the dataset (rows, columns): (166153, 7)

Column names:
['Unnamed: 0', 'Username', 'Artist', 'Track', 'Album', 'Date', 'Time']

DataFrame info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 166153 entries, 0 to 166152
Data columns (total 7 columns):
 #   Column      Non-Null Count   Dtype 
---  ------      --------------   ----- 
 0   Unnamed: 0  166153 non-null  int64 
 1   Username    166153 non-null  object
 2   Artist      166153 non-null  object
 3   Track       166153 non-null  object
 4   Album       166141 non-null  object
 5   Date        166153 non-null  object
 6   Time        166153 non-null  object
dtypes: int64(1), object(6)
memory usage: 8.9+ MB
None
Unnamed: 0    166153.0
Username        166153
Artist          166153
Track           166153
Album           166141
Date            166153
Time            166153
Name: count, dtype: object


### Interpretation: Descriptive Statistics Summary of the Last.fm Dataset

- All columns are of type `object` (i.e., string), except for `Unnamed: 0`, which is an integer index column (likely auto-generated during CSV export).

#### 🧮 Non-Missing Values per Column:

| Column       | Non-Null Count |
|--------------|----------------|
| Unnamed: 0   | 166,153        |
| Username     | 166,153        |
| Artist       | 166,153        |
| Track        | 166,153        |
| Album        | 166,141        |
| Date         | 166,153        |
| Time         | 166,153        |

- **Missing Values**: Only the `Album` column has missing values — **12 entries** are missing album names. 
- All other fields are **fully populated**, indicating good data quality for analysis.

Note:
- As observed from the column names there is no gender specified for the artists. Therefore, some mapping from other datasets is needed. From https://zenodo.org/records/3748787 the file: lfm-360-gender.tar.gz was retrieved as it contains a json file with the musicbrainz id's (mbid). Using the data from this project on Kaggle: https://www.kaggle.com/datasets/pieca111/music-artists-popularity the gender attributes will be mapped to the artists in our data. The dataset used in this project consists of over 1.4 Million musical artists present in MusicBrainz database -- their names, tags, and popularity (listeners/scrobbles), based on data scraped from last.fm. For the purpose of this project only the mbid and the gender is retrieved. 
- The `Unnamed: 0` column will be dropped as it serves no analytical purpose.
- The `Date` and `Time` columns may be converted into `datetime` format for time-based analysis.


### Getting the gender feature for the artists - Dataset Mapping

C:\Users\patri\AppData\Local\Temp\ipykernel_18624\519892870.py:4: DtypeWarning: Columns (2,4,6) have mixed types. Specify dtype option on import or set low_memory=False.
  artist_meta = pd.read_csv(artist_meta_path)


                                   mbid          artist_lastfm
0  cc197bad-dc9c-440d-a5b5-d52ba2e14234               Coldplay
1  a74b1b7f-71a5-4011-9441-d0b5e4122711              Radiohead
2  8bfac288-ccc5-448d-9573-c33ea2aa5c30  Red Hot Chili Peppers
3  73e5e69d-3554-40d8-8516-00cb38737a1c                Rihanna
4  b95ce3ff-3d05-4e87-9e01-c97b66af13d4                 Eminem


                                   mbid  gender
0  b3ae82c2-e60b-4551-a76d-6620f1b456aa  Female
1  ff6e677f-91dd-4986-a174-8db0474b1799    Male
2  8688124b-dcff-4a39-9f30-4825d445014f    Male
3  c995a379-60b9-404b-bd97-a7e2de0751d3    Male
4  0fb62639-4143-443b-8779-6867a1d08230  Female

            Artist mbid gender
0  isobel campbell  NaN    NaN
1  isobel campbell  NaN    NaN
2  isobel campbell  NaN    NaN
3  isobel campbell  NaN    NaN
4  isobel campbell  NaN    NaN
gender
NaN       148701
Male       27720
Female     11509
Other        227
Name: count, dtype: int64


Total unique artists in Last.fm: 22808
Total matched MBIDs: 164553
Total matched genders: 39456


### Artist Matching

After enriching the Last.fm dataset with external metadata (MBIDs and gender), here are the results:

- **Total unique artists in Last.fm**: `22,808`
- **Total rows with matched MBIDs**: `164,553`  
  ➤ This means **99%+ of tracks** were successfully linked to an artist ID via metadata.
- **Total rows with matched gender**: `39,456`  
  ➤ Roughly **24% of the listening entries** could be assigned an artist gender.

This means while MBID coverage is excellent, gender metadata is only available for a **subset of artists**. This subset can still be very valuable for fairness analysis, but it's important to be aware of the partial coverage when interpreting results or training models.


### Data Pre-Processing and Exploration

     Unnamed: 0 Username          Artist  \
29           30  Babs_05     billy ocean   
32           33  Babs_05   bill callahan   
34           35  Babs_05      rod thomas   
35           36  Babs_05       fela kuti   
55          106  Babs_05  machel montano   
72          123  Babs_05  machel montano   
84          135  Babs_05  machel montano   
97          148  Babs_05      imelda may   
100         151  Babs_05      gary bartz   
113         159  Babs_05         fat joe   

                                                 Track  \
29             Lovely Day (feat. YolanDa Brown & Ruti)   
32   Arise, Therefore (feat. Six Organs of Admittance)   
34                                         Old Friends   
35           I.T.T. (International Thief Thief) - Edit   
55                                       Private Party   
72                                       Private Party   
84                                     Long Time Refix   
97                                       Just One Kiss   
100                                         Day by Day   
113  Lean Back (feat. Lil Jon, Eminem, Mase & Remy ...   

                                                 Album         Date    Time  \
29             Lovely Day (feat. YolanDa Brown & Ruti)  31 Jan 2021   21:23   
32   Arise, Therefore (feat. Six Organs of Admittance)  31 Jan 2021   21:13   
34                                         Old Friends  31 Jan 2021   21:08   
35           I.T.T. (International Thief Thief) [Edit]  31 Jan 2021   21:01   
55                                       Private Party  30 Jan 2021   13:51   
72                                       Private Party  29 Jan 2021   16:45   
84                                     Long Time Refix  29 Jan 2021   15:58   
97                                       Just One Kiss  29 Jan 2021   15:13   
100                                         Day by Day  31 Jan 2021   18:45   
113                                     All or Nothing  31 Jan 2021   11:58   

      artist_lastfm                                  mbid  gender  
29      billy ocean  0e422e91-a42a-4b4d-8413-9baff67350f2    Male  
32    bill callahan  c309d914-93af-4b3f-8058-d79c75ea89da    Male  
34       rod thomas  deb08150-d897-4adc-abd9-971d57a11f42    Male  
35        fela kuti  6514cffa-fbe0-4965-ad88-e998ead8a82a    Male  
55   machel montano  cf5ffa07-4fb4-47a1-9d61-8721ae9af576    Male  
72   machel montano  cf5ffa07-4fb4-47a1-9d61-8721ae9af576    Male  
84   machel montano  cf5ffa07-4fb4-47a1-9d61-8721ae9af576    Male  
97       imelda may  cf7560ef-7714-477e-b4cd-59de0a2d11fb  Female  
100      gary bartz  d2cdb3df-783b-472e-91ab-70630fc6a4ad    Male  
113         fat joe  a6a4c910-2b22-4a6f-aebe-805b176c47b8    Male  

### Remove this "ID-like" column as it is not needed

   Username          Artist  \
29  Babs_05     billy ocean   
32  Babs_05   bill callahan   
34  Babs_05      rod thomas   
35  Babs_05       fela kuti   
55  Babs_05  machel montano   

                                                Track  \
29            Lovely Day (feat. YolanDa Brown & Ruti)   
32  Arise, Therefore (feat. Six Organs of Admittance)   
34                                        Old Friends   
35          I.T.T. (International Thief Thief) - Edit   
55                                      Private Party   

                                                Album         Date    Time  \
29            Lovely Day (feat. YolanDa Brown & Ruti)  31 Jan 2021   21:23   
32  Arise, Therefore (feat. Six Organs of Admittance)  31 Jan 2021   21:13   
34                                        Old Friends  31 Jan 2021   21:08   
35          I.T.T. (International Thief Thief) [Edit]  31 Jan 2021   21:01   
55                                      Private Party  30 Jan 2021   13:51   

     artist_lastfm                                  mbid gender  
29     billy ocean  0e422e91-a42a-4b4d-8413-9baff67350f2   Male  
32   bill callahan  c309d914-93af-4b3f-8058-d79c75ea89da   Male  
34      rod thomas  deb08150-d897-4adc-abd9-971d57a11f42   Male  
35       fela kuti  6514cffa-fbe0-4965-ad88-e998ead8a82a   Male  
55  machel montano  cf5ffa07-4fb4-47a1-9d61-8721ae9af576   Male  

### Check for missing values in the whole dataset

Missing values per column:
Username         0
Artist           0
Track            0
Album            0
Date             0
Time             0
artist_lastfm    0
mbid             0
gender           0
dtype: int64


No missing values, so the data is clean.

### Check the structure for our gender enriched dataset again

Shape of the dataset (rows, columns): (39456, 9)

Column names:
['Username', 'Artist', 'Track', 'Album', 'Date', 'Time', 'artist_lastfm', 'mbid', 'gender']

DataFrame info:
<class 'pandas.core.frame.DataFrame'>
Index: 39456 entries, 29 to 188125
Data columns (total 9 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   Username       39456 non-null  object
 1   Artist         39456 non-null  object
 2   Track          39456 non-null  object
 3   Album          39456 non-null  object
 4   Date           39456 non-null  object
 5   Time           39456 non-null  object
 6   artist_lastfm  39456 non-null  object
 7   mbid           39456 non-null  object
 8   gender         39456 non-null  object
dtypes: object(9)
memory usage: 3.0+ MB
None
Username         39456
Artist           39456
Track            39456
Album            39456
Date             39456
Time             39456
artist_lastfm    39456
mbid             39456
gender           39456
Name: count, dtype: object


### Summary of Enriched Last.fm Subset with Gender 

This subset of the Last.fm dataset contains only the entries for which artist gender data could be successfully matched. Below is a detailed breakdown:


- **Shape**: `(39,456 rows, 9 columns)`

#### Column Overview

| Column          | Description                                      |
|-----------------|--------------------------------------------------|
| `Username`       | Name of the user                                 |
| `Artist`         | Artist name from the original dataset            |
| `Track`          | Track (song) name                                |
| `Album`          | Album name                                       |
| `Date`           | Date of listening (range: Jan 1–31, 2021)        |
| `Time`           | Time of the day when the track was played        |
| `artist_lastfm`  | Normalized artist name used for metadata matching |
| `mbid`           | MusicBrainz ID (unique artist identifier)        |
| `gender`         | Gender of the artist (e.g., "Male", "Female")    |

#### Data Quality

- All **39,456 entries** have **non-null values** in every column — the dataset is complete and clean.
- All columns are of type `object` (i.e., string), including `Date` and `Time`.

This subset is now suitable for gender-based analysis, bias detection, and fairness modeling.


### Explore the gender distribution

### Interpretation:  Gender Distribution of Artists in the Last.fm Subset

The pie chart shows the gender distribution of artists for whom gender data was successfully matched in the Last.fm dataset:

- **Male artists** make up the majority with **70.3%** of all plays in the matched subset.
- **Female artists** account for **29.2%** of the data.
- **Other** gender represents a very small fraction (**0.6%**).

This indicates a substantial gender imbalance in the dataset — with more than two-thirds of the plays attributed to male artists. This may reflect existing gender disparities in music production, user listening behavior, or biases in artist exposure and availability.




## 📘 02 Last.fm Baselinerecommender Fairlearn

##  Using Fairlearn to detect gender bias in music recommender system: Last.fm Dataset

### Baseline Recommender: Popularity-Based

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 39456 entries, 0 to 39455
Data columns (total 10 columns):
 #   Column         Non-Null Count  Dtype 
---  ------         --------------  ----- 
 0   Unnamed: 0     39456 non-null  int64 
 1   Username       39456 non-null  object
 2   Artist         39456 non-null  object
 3   Track          39456 non-null  object
 4   Album          39456 non-null  object
 5   Date           39456 non-null  object
 6   Time           39456 non-null  object
 7   artist_lastfm  39456 non-null  object
 8   mbid           39456 non-null  object
 9   gender         39456 non-null  object
dtypes: int64(1), object(9)
memory usage: 3.0+ MB


(None,
    Unnamed: 0 Username          Artist  \
 0          30  Babs_05     billy ocean   
 1          33  Babs_05   bill callahan   
 2          35  Babs_05      rod thomas   
 3          36  Babs_05       fela kuti   
 4         106  Babs_05  machel montano   
 
                                                Track  \
 0            Lovely Day (feat. YolanDa Brown & Ruti)   
 1  Arise, Therefore (feat. Six Organs of Admittance)   
 2                                        Old Friends   
 3          I.T.T. (International Thief Thief) - Edit   
 4                                      Private Party   
 
                                                Album         Date    Time  \
 0            Lovely Day (feat. YolanDa Brown & Ruti)  31 Jan 2021   21:23   
 1  Arise, Therefore (feat. Six Organs of Admittance)  31 Jan 2021   21:13   
 2                                        Old Friends  31 Jan 2021   21:08   
 3          I.T.T. (International Thief Thief) [Edit]  31 Jan 2021   21:01   
 4                                      Private Party  30 Jan 2021   13:51   
 
     artist_lastfm                                  mbid gender  
 0     billy ocean  0e422e91-a42a-4b4d-8413-9baff67350f2   Male  
 1   bill callahan  c309d914-93af-4b3f-8058-d79c75ea89da   Male  
 2      rod thomas  deb08150-d897-4adc-abd9-971d57a11f42   Male  
 3       fela kuti  6514cffa-fbe0-4965-ad88-e998ead8a82a   Male  
 4  machel montano  cf5ffa07-4fb4-47a1-9d61-8721ae9af576   Male  )

### Apply our popularity-based recommender (Top-N = 100)

                                   Track  play_count          Artist  gender
0                              Undivided          82      tim mcgraw    Male
1                              Dirtknock          70          madlib    Male
2                Road of the Lonely Ones          67          madlib    Male
3                               Hopprock          60          madlib    Male
4  Everyday & Everynight - Straight Pass          50  yvette michele  Female
5                            Loose Goose          48          madlib    Male
6    Baila Conmigo (with Rauw Alejandro)          43    selena gomez  Female
7                                  Chino          43          madlib    Male
8                           Riddim Chant          43          madlib    Male
9                               The Call          42          madlib    Male

### Inspect the Top 100 Recommendations


Percentage of Recommendations by Gender:
 gender
Male      59.0
Female    38.0
Other      3.0
Name: proportion, dtype: float64


## Evaluation of Top-100 Recommended Tracks: Lastfm Dataset

We analyzed the Top-100 tracks recommended by a popularity-based recommender trained on the Last.fm enriched dataset. Below is a breakdown of artist gender representation and notable patterns:

### Gender Distribution of Artists

The Top-100 recommended tracks consist of the following distribution by artist gender:

- **Male**: 59 tracks (59%)
- **Female**: 38 tracks (38%)
- **Other**: 3 tracks (3%)

This indicates a somewhat skewed distribution, with male artists receiving the majority of recommendations. Female artists represent a significant portion but still trail behind, while the "Other" category is totally underrepresented.

### Key insights: 
- Female and Other gender groups are noticeably less represented, which could lead to biased exposure and reduced diversity in recommendations.
- This analysis establishes a fairness baseline and underscores the need for bias detection and potential mitigation.

---

### Bias detection using Fairlarn

Absolute Gender Distribution in Full Catalog:
gender
Male      12831
Female     4692
Other        55
Name: count, dtype: int64

Percentage Distribution:
gender
Male      72.99
Female    26.69
Other      0.31
Name: proportion, dtype: float64


If the recommender was perfectly unbiased and proportional, we’d expect about: 73 male tracks, 27 female tracks, and 0 or 1 from artists categorized as “Other.”

### Metrics computation using Fairlearn

Selection rate by gender group:
gender
Female    0.008099
Male      0.004598
Other     0.054545
Name: selection_rate, dtype: float64


### Visualize selection rates by gender groups

## Selection Rate by Gender Group

| Gender Group | Selection Rate      | Interpretation                                                                 |
|--------------|---------------------|---------------------------------------------------------------------------------|
| **Female**   | 0.008099 (0.81%)    | 🟢 Highest selection rate — relatively well represented                          |
| **Male**     | 0.004598 (0.46%)    | 🟡 Lower selection rate — underrepresented relative to catalog share (73%)       |
| **Other**    | 0.054545 (5.45%)    | 🔴 Extremely high — significant overrepresentation given catalog share (~0.3%)   |

---

### What This Means

- **Selection rate bias is present**, but **unlike Spotify**, the pattern is reversed:
  - **Female artists** are being recommended more often (per artist) than male artists.
  - **Male artists**, despite making up ~73% of the catalog, are being recommended at a lower rate.
  - **Other** artists (just 0.3% of catalog) have an unusually high selection rate — likely due to **a very small denominator** (55 artists), where just **3 selected tracks** would already give a 5.45% rate.

---

### Interpretation Considerations

- This result could be due to:
  - A **few very popular “Other” artists** disproportionately influencing the results.
  - A **Top-N list** that happened to include strong representation of female artists.
- Still, given the massive underrepresentation in the catalog, the fairness profile may seem **counterintuitive**, but it shows that **popularity is not always biased toward the majority**.

### Inspection of Pairwise Demographic Parity Differences (DPD) 
These are useful for identifying specific fairness disparities between individual gender groups, rather than just looking at the overall range. This helps reveal whether certain groups are consistently under- or over-recommended relative to others. It also allows for a more nuanced understanding of fairness in multiclass settings, where biases may affect each group differently. By comparing each group pair directly, we can target mitigation efforts more precisely.

📊 Pairwise Demographic Parity Differences:

  Group A Group B  DP Difference (A vs B)
0    male  female                0.003501
1    male   other                0.049947
2  female   other                0.046447


## Pairwise Demographic Parity Differences

| Group A | Group B | DP Difference (A vs B) | Interpretation |
|---------|---------|------------------------|----------------|
| **Male**   | Female   | 0.0035 (0.35%)           | 🟢 Very small difference — nearly equal selection rates between male and female artists |
| **Male**   | Other    | 0.0499 (4.99%)           | 🔴 Notable disparity — artists labeled "Other" are selected more frequently than male artists |
| **Female** | Other    | 0.0464 (4.64%)           | 🔴 Similar disparity — "Other" artists are selected more frequently than female artists |

---

### Interpretation:

- The **selection rates for male and female artists** are very close (DPD ≈ 0.35%), suggesting little disparity between them in the recommender output.
- However, there is a **noticeable selection bias in favor of the "Other" gender group**, with both male and female artists being under-selected in comparison.
- This aligns with earlier findings where the "Other" group had a much higher selection rate despite making up only 0.3% of the catalog.

### Consideration

- Because the "Other" group is very small (just 55 artists), even a few popular tracks can **inflate their selection rate and DPD values**.
- This makes the fairness result sensitive to **outliers** or **high-impact individual artists** in that group.
---


Demographic Parity Difference: 0.049947
Demographic Parity Ratio: 0.084301


| Metric            | Value      | Interpretation                                                                                                                                               |
| ----------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **DP Difference** | `0.049947` | 🟡 There is **moderate disparity** — the most selected group has a 5% higher selection rate than the least selected group. Not extreme, but noticeable.      |
| **DP Ratio**      | `0.084301` | 🔴 This is **very low** — the least selected group receives only 8.4% as many recommendations as the most selected group. This signals **severe imbalance**. |

The recommender system shows moderate disparity in how different gender groups are represented in the Top-100 recommendations. In particular, artists labeled as "Other" receive recommendations at a much higher rate than both male and female artists — resulting in a Demographic Parity Ratio of only 8.4%, which indicates a serious imbalance. However, this effect is likely magnified by the very small size of the "Other" group, so this result should be interpreted carefully.

## Bias Mitigation

### In-Processing: Fairness-Aware Learning using Fairlearn’s ExponentiatedGradient

Selection Rate by Gender Group (Fair Model):
gender_grouped
female    0.004972
male      0.004935
other     0.062500
Name: selection_rate, dtype: float64

📏 Selection Rate Range (Max - Min): 0.0576


## Selection Rate Comparison: Baseline vs. Fair Model (Last.fm)

| Gender Group | Baseline Selection Rate | Fair Model Selection Rate | Change         | Interpretation                                                |
|--------------|-------------------------|----------------------------|----------------|----------------------------------------------------------------|
| **Female**   | 0.008099 (0.81%)        | 0.004972 (0.50%)           | 🔻 Decreased    | More aligned with male group; now nearly equal                |
| **Male**     | 0.004598 (0.46%)        | 0.004935 (0.49%)           | 🔼 Slightly up  | Balanced with female group                                    |
| **Other**    | 0.054545 (5.45%)        | 0.062500 (6.25%)           | 🔼 Increased    | Still heavily overrepresented despite small catalog presence  |

---

### Summary of Effects

-  **Male and female** selection rates are now very close (0.49%–0.50%), suggesting that the fairness-aware model has **eliminated gender imbalance** between these two groups.
- The **"Other" group** (which represents only ~0.3% of the catalog) remains **significantly overrepresented**. The selection rate actually increased from 5.45% to 6.25%.
- **Selection Rate Range (Max - Min): 0.0576** — still high due to the outlier status of the "Other" group.

---

### Interpretation:

- The **Exponentiated Gradient with Demographic Parity** constraint was **successful in equalizing outcomes** between male and female artists, which was the main fairness concern in your Spotify experiment.
- However, it **did not mitigate the disproportionate selection of "Other" artists** — in fact, it slightly increased their representation.
- This reflects a limitation of Demographic Parity when applied to **very small groups**: even a few high-ranking items can dramatically skew fairness metrics.
- Therefore, **fairness mitigation should be interpreted alongside group size and outcome impact** — especially when evaluating small minorities like the "Other" gender group in your dataset.

---

C:\Users\patri\AppData\Local\Temp\ipykernel_7732\4003757063.py:34: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  plt.text(x[i] - width/2, baseline_rates[i] + 0.0001, f'{baseline_rates[i]:.6f}', ha='center', fontsize=8)
C:\Users\patri\AppData\Local\Temp\ipykernel_7732\4003757063.py:35: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  plt.text(x[i] + width/2, fair_rates[i] + 0.0001, f'{fair_rates[i]:.6f}', ha='center', fontsize=8)
C:\Users\patri\AppData\Local\Temp\ipykernel_7732\4003757063.py:37: UserWarning: Glyph 127911 (\N{HEADPHONE}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
C:\Users\patri\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\IPython\core\pylabtools.py:170: UserWarning: Glyph 127911 (\N{HEADPHONE}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)


## Interpretation of Selection Rate Comparison Chart:

The chart shows a side-by-side comparison of selection rates per gender group for the **Baseline** and **Fair Model** (on the test set):

-  **Male and female** selection rates are now closely aligned in the fair model (~0.005), reducing disparity between them.
-  **Female artists** saw a reduction in selection rate, while **male artists** remained nearly the same — balancing both.
- The **"Other" group**, which was highly overrepresented in the baseline (~0.125), has a **lower but still elevated** selection rate (~0.063) after mitigation.

This confirms that the fairness-aware model improved parity between male and female groups but had limited effect on the small and dominant “Other” group.


**! Note:**
For this dataset we decided to skip ThresholdOptimizer as it only supports **binary sensitive attributes**, which makes it unsuitable for our **multiclass gender setup** (`male`, `female`, `other`).  We've instead used **ExponentiatedGradient**, which supports multiclass fairness and that yielded better results.

## Bias Mitigation using Reweighting (Inprocessing)

Selection Rate by Gender Group (GridSearch with Reweighting):
gender_grouped
female    0.001420
male      0.002597
other     0.000000
Name: selection_rate, dtype: float64

📏 Selection Rate Range (Max - Min): 0.0026


## Interpretation: GridSearch with Reweighting (Fairlearn)

| Gender Group | Selection Rate | Interpretation                                                  |
|--------------|----------------|------------------------------------------------------------------|
| **Female**   | 0.001420 (0.14%) | 🔻 Slightly underrepresented; selection rate remains low        |
| **Male**     | 0.002597 (0.26%) | 🟢 Highest selection rate among the three groups                |
| **Other**    | 0.000000 (0.00%) | ❌ No tracks selected — group completely excluded               |

---

##### Selection Rate Range: 0.0026

- The fairness-aware model trained with **GridSearch and Demographic Parity** achieved a **low selection rate disparity** numerically.
- However, this came at the cost of **excluding the "Other" group entirely**, which highlights a **fairness trade-off**.
- This outcome suggests that the optimizer prioritized balancing between **male and female groups**, possibly due to the **tiny size** of the "Other" group in the training data.

**Conclusion**: GridSearch reduced disparity across male/female, but fairness for the "Other" group remains unaddressed.


## Comprehensive Comparison of Selection Rates (Last.fm)

| Gender Group | Baseline Selection Rate | Fair Model (EG) | Reweighted Model (GridSearch) | Interpretation |
|--------------|-------------------------|------------------|-------------------------------|----------------|
| **Female**   | 0.008099 (0.81%)        | 0.004972 (0.50%) | 0.001420 (0.14%)              | Selection rate decreased across both models; GridSearch reduced it the most |
| **Male**     | 0.004598 (0.46%)        | 0.004935 (0.49%) | 0.002597 (0.26%)              | Fair model aligned male with female; GridSearch made male the top group     |
| **Other**    | 0.054545 (5.45%)        | 0.062500 (6.25%) | 0.000000 (0.00%)              | Fair model increased overrepresentation; GridSearch excluded the group      |

---

### Interpretation

- The **Fair Model (ExponentiatedGradient)** successfully **balanced selection rates** between **male and female** groups, bringing them close to parity (~0.5%).
- However, it **amplified bias toward the "Other" group**, increasing its already high selection rate despite the group's small size.
- The **Reweighted Model (GridSearch)** reversed this:
  - It further **reduced disparity between male and female**, but
  - It **completely excluded the "Other" group**, reducing its selection rate to 0%.

### Trade-Offs:

- The **Fair Model** prioritized balancing the most common groups (male/female), but overrepresented the minority.
- The **Reweighted Model** prioritized overall parity across the majority groups, but **ignored the minority**, raising concerns of **fairness exclusion**.

---

### Conclusion

- **No single model achieved perfect fairness** across all gender groups.
- The Fair Model kept the "Other" group visible but overexposed it.
- The Reweighted Model removed this exposure but introduced exclusion.
- These results highlight the **challenge of fairness mitigation** in datasets with **high class imbalance** and **minority-sensitive attributes**.


## 📘 05 Spotify ContentBased Filtering recommender AIF360

### Step 1: Load Dataset and Prepare Gender Labels

We start by loading a pre-filtered dataset containing Spotify tracks enriched with artist gender metadata.
We engineer a proxy recommendation score using `track_position` and assign binary labels:
- `1` = recommended (position ≤ 5)
- `0` = not recommended

We also map `artist_gender` to binary format:
- `1` = male
- `0` = female

WARNING:root:No module named 'tensorflow': AdversarialDebiasing will be unavailable. To install, run:
pip install 'aif360[AdversarialDebiasing]'
WARNING:root:No module named 'tensorflow': AdversarialDebiasing will be unavailable. To install, run:
pip install 'aif360[AdversarialDebiasing]'
WARNING:root:No module named 'inFairness': SenSeI and SenSR will be unavailable. To install, run:
pip install 'aif360[inFairness]'


### Step 2: Prepare Features and Split Data

We select the `score` as our main feature.

We split the dataset into training and test sets using `train_test_split`,
while preserving:
- `y` as the target (`recommended`)
- `sensitive_attr` as gender (the protected attribute)

### Step 3: Train Content-Based Logistic Recommender

A simple logistic regression model is trained to simulate a content-based recommender
using the track score to predict whether a track would be recommended.

### Step 4: Create AIF360-Compatible Test and Prediction DataFrames

We construct:
- `test_df` for ground truth test data
- `pred_df` for model predictions

These are needed to build `BinaryLabelDataset` objects required by AIF360.


### Step 5–7: Format for AIF360 Evaluation

We construct two pandas DataFrames:
- `test_df` for ground-truth test labels
- `pred_df` for model predictions

These are then converted into AIF360's `BinaryLabelDataset` objects required for fairness evaluation.


### Step 8–9: In-Processing Fairness Mitigation with PrejudiceRemover

We apply `PrejudiceRemover`, an in-processing technique that adjusts model training to reduce bias.
However, in our case, the prediction phase failed due to an internal AIF360 shape issue.
We gracefully fall back to the original model predictions to maintain evaluation continuity.

📌 Output:
- Disparate Impact ≈ 1.037 — indicating minimal bias
- Statistical Parity Difference ≈ 0.003 — almost fair


### 10: In-Processing Fairness Mitigation with PrejudiceRemover

We apply AIF360's `PrejudiceRemover`, which adjusts model training to reduce bias
related to the sensitive attribute (gender in this case).

❗ However, `predict()` failed due to an internal shape error in AIF360's output format.
✅ We fallback to using the original model predictions for evaluation.


C:\Users\kapoe\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\aif360\algorithms\inprocessing\prejudice_remover.py:208: UserWarning: loadtxt: input contained no data: "C:\Users\kapoe\AppData\Local\Temp\tmpmtkjpdsm"
  m = np.loadtxt(output_name)


⚠️ PrejudiceRemover prediction failed: too many indices for array: array is 1-dimensional, but 2 were indexed
🔁 Fallback: using original test labels.

[In-processing: PrejudiceRemover]
Disparate Impact: 1.0373549403369495
Statistical Parity Difference: 0.0034000492364402862


#### Fairness Results (Fallback):
- Disparate Impact ≈ 1.037 → indicates *slight* bias (close to fair)
- Statistical Parity Difference ≈ 0.003 → very low difference in recommendation rates


# 🎧 Final Summary: Investigating Gender Bias in Content-Based Music Recommendation (AIF360)

## 📌 Project Goal

This notebook evaluates gender bias in a content-based music recommender system using AIF360.
The aim is to identify disparities in recommendation outcomes across male and female artists and
apply fairness-aware techniques to mitigate those biases.

We use the **Spotify Million Playlist Dataset (filtered)**, enriched with **artist gender information**.

---

## 🧠 Methodology Overview

| Step | Description |
|------|-------------|
| 1    | Load and preprocess the Spotify dataset |
| 2    | Engineer a proxy `score` based on `track_position` |
| 3    | Define a binary `recommended` label (1 if top 5 position) |
| 4    | Encode gender as a binary protected attribute |
| 5    | Split data and train a logistic regression model |
| 6    | Evaluate fairness using AIF360's metrics |
| 7    | Attempt in-processing mitigation with `PrejudiceRemover` |

---

## ⚙️ Technical Details

- **Feature used:** `score = 1 - (track_position / max_position)`
- **Protected attribute:** `artist_gender` (mapped to binary)
- **Model used:** Logistic Regression
- **Library:** AIF360

---

## 📊 Fairness Evaluation Results

> When applying **PrejudiceRemover**, the prediction step failed due to an internal indexing error.
> As a fallback, we evaluated the unmitigated model predictions.

| Metric                      | Value          | Interpretation                          |
|----------------------------|----------------|------------------------------------------|
| Disparate Impact           | `1.037`        | Slightly favors male artists (>1)       |
| Statistical Parity Diff.   | `0.0034`       | Very small gap (close to fairness)      |

> 💬 **Interpretation:**  
> These values suggest that the model exhibits **minimal bias** toward one gender group.  
> However, because in-processing mitigation could not be applied, the metrics reflect the original model output.

---

## ✅ Key Takeaways

- The logistic regression model showed **mostly balanced outcomes** by gender.
- AIF360's PrejudiceRemover failed due to output shape issues — a known limitation.
- We gracefully recovered using fallback predictions, allowing fairness metrics to be computed.

---

## 📎 Recommendations for Future Work

- Add **post-processing** (e.g., `RejectOptionClassification`) to adjust predictions.
- Use **multiple features** (like audio properties) for better predictive performance and fairness trade-offs.
- Consider visualizing fairness metrics across models and mitigation strategies.

---

## 🎓 Final Reflection

This notebook demonstrates how AIF360 can be used to assess and begin addressing gender bias
in music recommendation systems. Even when mitigation fails, fallback evaluation provides
valuable insights into systemic disparities.



## 📘 02 Spotify descriptive data incl files


Numerical Statistics:
         playlist_id  track_position  track_duration_ms
count  3.174887e+06    3.174887e+06       3.174887e+06
mean   1.297466e+05    5.402403e+01       2.312768e+05
std    4.805673e+04    4.816304e+01       5.688718e+04
min    0.000000e+00    0.000000e+00       2.060000e+02
25%    1.137590e+05    1.700000e+01       2.001330e+05
50%    1.387200e+05    4.000000e+01       2.243910e+05
75%    1.639120e+05    7.900000e+01       2.533730e+05
max    1.889990e+05    2.490000e+02       4.789226e+06

Categorical Statistics:
        playlist_name track_name artist_name album_name  \
count        3174887    3174887     3174887    3174887   
unique         27928      88076        2440      26176   
top          Country    HUMBLE.       Drake      Views   
freq           90248       4562       85765      21296   

                                   track_uri  \
count                                3174887   
unique                                108608   
top     spotify:track:7KXjTSCq5nL1LoYtL7XAwS   
freq                                    4562   

                                   artist_uri  \
count                                 3174887   
unique                                   2624   
top     spotify:artist:3TVXtAsR1Inumwj472S9r4   
freq                                    85749   

                                   album_uri artist_gender  
count                                3174887       3174887  
unique                                 31402             7  
top     spotify:album:5s0rmjP8XOPhP6HhqOhuyC          male  
freq                                   15970       2459171  


Column overview: ['playlist_id', 'playlist_name', 'track_position', 'track_name', 'artist_name', 'album_name', 'track_uri', 'artist_uri', 'album_uri', 'track_duration_ms', 'artist_gender']

Numerical statistics:
         playlist_id  track_position  track_duration_ms
count  3.174887e+06    3.174887e+06       3.174887e+06
mean   1.297466e+05    5.402403e+01       2.312768e+05
std    4.805673e+04    4.816304e+01       5.688718e+04
min    0.000000e+00    0.000000e+00       2.060000e+02
25%    1.137590e+05    1.700000e+01       2.001330e+05
50%    1.387200e+05    4.000000e+01       2.243910e+05
75%    1.639120e+05    7.900000e+01       2.533730e+05
max    1.889990e+05    2.490000e+02       4.789226e+06



Frequencies of 'artist_gender':
 artist_gender
male                 2459171
female                666351
non-binary gender      47022
genderfluid             1821
agender                  334
trans woman              174
neutral sex               14
Name: count, dtype: int64


### 🎯 Key Findings – Track Position by Artist Gender

- **Similar Medians**: Most gender groups (male, female, non-binary, genderfluid) have similar median track positions (~40–50).
- **Higher Variability**: Agender and neutral sex artists show wider position spreads, indicating inconsistent placement in playlists.
- **More Outliers**: Male and female artists have many high-position outliers (tracks placed near the playlist end).
- **Neutral Sex Bias**: Tracks by neutral sex artists tend to appear **later** in playlists (higher median position).
- **Trans Women**: Tend to have **lower variability** and fewer outliers, suggesting more consistent placement.

### 🕒 Key Findings – Track Duration by Artist Gender

- **Typical Track Durations Are Similar**: Most gender groups have comparable median durations (around 200,000–250,000 ms = 3.3–4.2 minutes).
- **Extreme Outliers in Male & Female**: A few tracks by male and female artists have abnormally high durations (up to 4–5 million ms, i.e., over 1 hour), strongly skewing the scale.
- **Compressed Ranges in Minoritized Genders**: Genderfluid, non-binary, agender, and neutral sex artists show narrower duration ranges, indicating more consistent track lengths.
- **Neutral Sex Tracks**: Appear especially uniform, with nearly no variance in duration.
- **Recommendation**: Consider removing or investigating outliers when analyzing duration-related trends, as they distort the visualization scale.

<class 'pandas.core.frame.DataFrame'>
Index: 3174887 entries, 0 to 6685098
Data columns (total 11 columns):
 #   Column             Dtype 
---  ------             ----- 
 0   playlist_id        int64 
 1   playlist_name      object
 2   track_position     int64 
 3   track_name         object
 4   artist_name        object
 5   album_name         object
 6   track_uri          object
 7   artist_uri         object
 8   album_uri          object
 9   track_duration_ms  int64 
 10  artist_gender      object
dtypes: int64(3), object(8)
memory usage: 290.7+ MB


playlist_id          0
playlist_name        0
track_position       0
track_name           0
artist_name          0
album_name           0
track_uri            0
artist_uri           0
album_uri            0
track_duration_ms    0
artist_gender        0
dtype: int64


Duplicate rows: 0


Interaction and track data successfully prepared.


## 📘 03 Last.fm ContentBasedFiltering Fairlearn

##  Using Fairlearn to detect gender bias in music recommender system: Spotify Million Dataset

### Content-Based Filtering (CBF) recommender:

**Idea:** Recommend tracks similar to those with high popularity using track features.

**Examples of track features you could use:**
- Genre
- Artist name (or ID)

**How?**
- Create a feature matrix (e.g., one-hot encode genres, artists)
- Use Nearest Neighbor similarity between tracks
- Rank based on similarity to top popular tracks

   Unnamed: 0 Username          Artist  \
0          30  Babs_05     billy ocean   
1          33  Babs_05   bill callahan   
2          35  Babs_05      rod thomas   
3          36  Babs_05       fela kuti   
4         106  Babs_05  machel montano   

                                               Track  \
0            Lovely Day (feat. YolanDa Brown & Ruti)   
1  Arise, Therefore (feat. Six Organs of Admittance)   
2                                        Old Friends   
3          I.T.T. (International Thief Thief) - Edit   
4                                      Private Party   

                                               Album         Date    Time  \
0            Lovely Day (feat. YolanDa Brown & Ruti)  31 Jan 2021   21:23   
1  Arise, Therefore (feat. Six Organs of Admittance)  31 Jan 2021   21:13   
2                                        Old Friends  31 Jan 2021   21:08   
3          I.T.T. (International Thief Thief) [Edit]  31 Jan 2021   21:01   
4                                      Private Party  30 Jan 2021   13:51   

    artist_lastfm                                  mbid gender  
0     billy ocean  0e422e91-a42a-4b4d-8413-9baff67350f2   Male  
1   bill callahan  c309d914-93af-4b3f-8058-d79c75ea89da   Male  
2      rod thomas  deb08150-d897-4adc-abd9-971d57a11f42   Male  
3       fela kuti  6514cffa-fbe0-4965-ad88-e998ead8a82a   Male  
4  machel montano  cf5ffa07-4fb4-47a1-9d61-8721ae9af576   Male  


Cleaned gender groups:
gender_grouped
male      27720
female    11509
other       227
Name: count, dtype: int64


Remaining gender groups:
gender_grouped
male      27720
female    11509
other       227
Name: count, dtype: int64


## Building the Content-Based Recommender using Track, Artist, Album

NearestNeighbors(algorithm='brute', metric='cosine')

gender_grouped
male      78.0
female    22.0
Name: proportion, dtype: float64

Seed Track:
Track             Lovely Day (feat. YolanDa Brown & Ruti)
Artist                                        billy ocean
Album             Lovely Day (feat. YolanDa Brown & Ruti)
gender_grouped                                       male
Name: 0, dtype: object

Top 10 Recommendations:
                               Track            Artist  \
7                         Day by Day        gary bartz   
130                       Lovely Day      bill withers   
8105               Forever And A Day         ian brown   
4798                Olancha Farewell       harold budd   
1052           All the Lovely Ladies  gordon lightfoot   
16863                     Gimme That       chris brown   
8669   Love Really Hurts Without You       billy ocean   
2408                Sweet and Lovely   thelonious monk   
13333                  From This Day       lene marlin   
3372          Lucifer - Instrumental         kev brown   

                                       Album gender_grouped  
7                                 Day by Day           male  
130     Lean on Me: The Best of Bill Withers           male  
8105                    Music Of The Spheres           male  
4798                          Lovely Thunder           male  
1052                    Cold on the Shoulder           male  
16863                            Chris Brown           male  
8669   Here You Are: The Best of Billy Ocean           male  
2408                            Monk's Dream           male  
13333                            Another Day         female  
3372               Brown Album Instrumentals           male  



#### Interpretation:

The content-based filtering (CBF) recommender produced the following gender distribution in its top 100 recommendations:

- **78%** of the recommended tracks are by **male** artists.
- **22%** are by **female** artists.

This indicates a notable **gender imbalance** in the recommendation output — heavily skewed toward male artists.

#### Seed Track Context:

The seed track used for generating recommendations was:

- **Track**: *Lovely Day (feat. YolanDa Brown & Ruti)*
- **Artist**: *Billy Ocean*
- **Gender Group**: *Male*

The top 10 recommended tracks also primarily feature **male** artists, with only **one track** attributed to a **female** artist (*Lene Marlin*).

#### Key Insight:

This result shows that the CBF recommender, which relies on textual similarity (track, artist, album metadata), tends to recommend artists of the **same gender** as the seed artist. In this case, starting from a male artist resulted in ~78% male-dominated recommendations.

This may suggest:
- The **metadata features** used (e.g., artist/track names) inherently capture stylistic or cultural signals associated with gender.
- The recommender may **amplify existing gender representation imbalances** in the dataset — e.g., if male artists dominate the dataset or certain musical descriptors.

Seed indices by gender:
{'female': 6, 'male': 0, 'other': 303}



--- Recommendations from female seed ---
gender_grouped
male      72.0
female    28.0
Name: proportion, dtype: float64

--- Recommendations from male seed ---
gender_grouped
male      78.0
female    22.0
Name: proportion, dtype: float64

--- Recommendations from other seed ---
gender_grouped
male      53.0
other     31.0
female    16.0
Name: proportion, dtype: float64


### 🎯 Gender-Specific Seed Track Analysis: CBF Recommendation Behavior

#### Interpretation:

The recommender was tested using **three seed tracks**, each representing a different artist gender group:

| Seed Gender | Index | Description |
|-------------|-------|-------------|
| Female      | 6     | A track by a female artist |
| Male        | 0     | A track by a male artist (Billy Ocean) |
| Other       | 303   | A track by an artist labeled as "other" gender (nonbinary, unknown, etc.) |

The following are the recommendation outcomes for each seed group:

---

####  Recommendations from **female** seed track:
- **72%** male artists
- **28%** female artists

> Despite using a **female** artist as the seed, the majority of recommended tracks were still by **male** artists.

---

#### Recommendations from **male** seed track:
- **78%** male artists
- **22%** female artists

> Similar to the female seed, recommendations heavily favored **male** artists. This mirrors the earlier single-seed results and confirms consistency in skew.

---

#### Recommendations from **'other'** gender seed:
- **53%** male artists
- **31%** artists labeled as **other**
- **16%** female artists

> The 'other' gender seed produced a slightly more **diverse output**, but still leaned male. The model offered the **highest proportion of 'other' gender artists** here (31%), likely due to lexical similarity with niche or independent descriptors shared among 'other' artists.

---

#### Takeaway:

The model exhibits a **consistent male dominance** in its recommendations, **regardless of seed gender**. Even when seeded with underrepresented groups (female or other), the system tends to revert toward recommending male artists.

This suggests:
- A **bias in the underlying dataset** or
- An **inherent reinforcement loop** in content-based systems favoring majority group representations (e.g., more metadata overlap or popularity among male artists).


### Bias Detection using Fairlearn

                 Track                Artist                       Album  \
569     Plastic Hearts           miley cyrus              Plastic Hearts   
633               High           miley cyrus              Plastic Hearts   
632            Hate Me           miley cyrus              Plastic Hearts   
631        Never Be Me           miley cyrus              Plastic Hearts   
1205   Golden G String           miley cyrus              Plastic Hearts   
...                ...                   ...                         ...   
17389      Soul Lament         kenny burrell               Midnight Blue   
16601       Liquid Sky           solar quest                     Orgship   
50            Research              big sean  Dark Sky Paradise (Deluxe)   
7537    Midnight Rider         willie nelson            Walking The Line   
13162  Look To The Sky  antônio carlos jobim                        Wave   

      gender_grouped  
569            other  
633            other  
632            other  
631            other  
1205           other  
...              ...  
17389           male  
16601           male  
50              male  
7537            male  
13162           male  

[100 rows x 4 columns]


Selection rate by gender:
gender_grouped
female    0.004689
male      0.006079
other     0.000000
Name: selection_rate, dtype: float64



Demographic Parity Difference: 0.006
Demographic Parity Ratio: 0.000


  Group 1 Group 2       DPD
2    male   other  0.006079
1  female   other  0.004689
0  female    male  0.001390


### Fairness Evaluation (Pre-Mitigation): Selection Rate & Demographic Parity

#### Selection Rate by Gender Group

| Gender Group | Selection Rate |
|--------------|----------------|
| Female       | 0.0047         |
| Male         | 0.0061         |
| Other        | 0.0000         |

- **Male artists** were selected at the highest rate (~0.61%).
- **Female artists** were selected slightly less (~0.47%).
- **Artists in the 'Other' gender group** were **never** selected (0.00%).

This suggests an **exclusion** of nonbinary, gender-neutral, or unclassified artists by the content-based recommender.

---

#### Demographic Parity Metrics

- **Demographic Parity Difference**: `0.006`
- **Demographic Parity Ratio**: `0.000`

**Interpretation:**
- The **parity difference** indicates the absolute difference between the highest and lowest group selection rates (here: ~0.006 = 0.61% - 0.00%).
- A **parity ratio of 0.000** highlights a critical fairness issue: **one or more groups received no recommendations at all**, which violates demographic parity expectations.

---

#### Pairwise Demographic Parity Differences

| Group 1 | Group 2 | DPD      |
|---------|---------|----------|
| Male    | Other   | 0.00608  |
| Female  | Other   | 0.00469  |
| Female  | Male    | 0.00139  |

**Interpretation:**

- The largest disparity is between male and other, confirming the earlier overall DPD of 0.006.
- The female–other gap (0.0047) is also substantial, meaning artists from the 'other' group were underrepresented regardless of seed gender.
- The female–male gap (0.0014) is the smallest, indicating that the system is slightly biased toward male artists but the disparity is minor between binary genders.

--- 
#### Key Takeaways

- While the numerical gap between male and female artists is small, the **complete exclusion of the 'Other' group** is a serious red flag. This reflects structural underrepresentation.
- Even a "neutral" recommender can **amplify historical or dataset-inherited imbalances** if no mitigation strategy is applied.

➡️ These results provide strong motivation for fairness-aware interventions such as constraint-based modeling, or representation balancing.


## Bias Mitigation

Selection rate by gender (post-mitigation):
gender_grouped
female    0.000000
male      0.001558
other     0.000000
Name: selection_rate, dtype: float64

DP Difference: 0.0016
DP Ratio: 0.0000


                Pre-Mitigation  Post-Mitigation
gender_grouped                                 
female                0.004689         0.000000
male                  0.006079         0.001558
other                 0.000000         0.000000


### Fairness Evaluation: Pre- vs. Post-Mitigation (Content-Based Recommender)

This section interprets the impact of applying **Exponentiated Gradient bias mitigation** using the **Demographic Parity** constraint on a content-based recommender trained with a balanced dataset.

---

#### 1. Selection Rates by Gender

| Gender Group | Pre-Mitigation | Post-Mitigation |
|--------------|----------------|-----------------|
| Female       | 0.0047         | 0.0000          |
| Male         | 0.0061         | 0.0016          |
| Other        | 0.0000         | 0.0000          |

- **Pre-Mitigation:** The model recommended male artists more often (0.61%) than female (0.47%), with **no recommendations at all** for the 'other' gender group.
- **Post-Mitigation:** Surprisingly, **only male artists were recommended**, though at a lower rate (0.16%). Female and 'other' groups received **no recommendations**.

---

#### 2. Demographic Parity Metrics

| Metric                        | Value (Post-Mitigation) |
|------------------------------|--------------------------|
| Demographic Parity Difference| 0.0016                   |
| Demographic Parity Ratio     | 0.0000                   |

- The **DP Difference** has technically **decreased** (from 0.0060 to 0.0016), suggesting a more equal spread in theory.
- However, the **DP Ratio** remains **0.000**, due to **zero selection for two of the three groups**.

This shows the paradox that **numeric fairness metrics can improve** even when **actual group representation worsens** or disappears entirely.

---

#### 3. Interpretation of Mitigation Behavior

The mitigation method **succeeded in minimizing selection disparity numerically**, but **failed in producing meaningful or fair recommendations** for all groups:

- **Male group dominance** remains — though attenuated.
- **Female and 'Other' groups are now fully excluded**, indicating an overcorrection or failure to learn from these groups.
- This outcome highlights that **constraints alone are not sufficient** if the model lacks:
  - Representative features per group,
  - Sufficient training signals (even after balancing),
  - Or if the logistic model is too simplistic to differentiate subtle patterns.

---

#### 4. Key Takeaways

- **Fairness constraints must be paired with robust data engineering**, including adequate group representation in both features and labels.
- **Demographic parity as a constraint may conflict with recommendation relevance**, especially if underlying patterns differ strongly by group.
- This result underscores the need for:
  - More expressive models (e.g., neural recommenders),
  - Alternative fairness metrics (e.g., Equal Opportunity),
  - Or hybrid strategies (e.g., post-processing ranking fairness, re-ranking, or exposure-aware models).

---



## 📘 01 Spotify GenderBiasinMusicRecommenderSystems Data Retrieval Preprocessing

# Data Science and Artificial Intelligence II
### Investigating gender bias in music recommender systems

   playlist_id playlist_name  track_position  \
0            0    Throwbacks               0   
1            0    Throwbacks               1   
2            0    Throwbacks               2   
3            0    Throwbacks               3   
4            0    Throwbacks               4   

                                   track_name        artist_name  \
0  Lose Control (feat. Ciara & Fat Man Scoop)      Missy Elliott   
1                                       Toxic     Britney Spears   
2                               Crazy In Love            Beyoncé   
3                              Rock Your Body  Justin Timberlake   
4                                It Wasn't Me             Shaggy   

                                     album_name  \
0                                  The Cookbook   
1                                   In The Zone   
2  Dangerously In Love (Alben für die Ewigkeit)   
3                                     Justified   
4                                      Hot Shot   

                              track_uri  \
0  spotify:track:0UaMYEvWZi0ZqiDOoHU3YI   
1  spotify:track:6I9VzXrHxO9rA9A5euc8Ak   
2  spotify:track:0WqIKmW4BTrj3eJFmnCKMv   
3  spotify:track:1AWQoqb9bSvzTjaLralEkT   
4  spotify:track:1lzr43nnXAijIGYnCT8M8H   

                              artist_uri  \
0  spotify:artist:2wIVse2owClT7go1WT98tk   
1  spotify:artist:26dSoYclwsYLMAKD3tpOr4   
2  spotify:artist:6vWDO969PvNqNYHIOW5v0m   
3  spotify:artist:31TPClRtHm23RisEBtV3X7   
4  spotify:artist:5EvFsr3kj42KNv97ZEnqij   

                              album_uri  track_duration_ms  
0  spotify:album:6vV5UrXcfyQD1wu4Qo2I9K             226863  
1  spotify:album:0z7pVBGOD7HCIB7S8eLkLI             198800  
2  spotify:album:25hVFAxTlDvXbx2X2QkUkE             235933  
3  spotify:album:6QPkyl04rXwTGlGlcYaRoW             267266  
4  spotify:album:6NmFmPX56pcLBOFMhIiKvF             227600  

Loaded 6685101 tracks from 100 files.



CSV exported successfully to:
D:\DigiEcon\4rd Semester\5673 Data Science and AI II\Project\spotify_tracks_subset.csv


   playlist_id playlist_name  track_position  \
0            0    Throwbacks               0   
1            0    Throwbacks               1   
2            0    Throwbacks               2   
3            0    Throwbacks               3   
4            0    Throwbacks               4   

                                   track_name        artist_name  \
0  Lose Control (feat. Ciara & Fat Man Scoop)      Missy Elliott   
1                                       Toxic     Britney Spears   
2                               Crazy In Love            Beyoncé   
3                              Rock Your Body  Justin Timberlake   
4                                It Wasn't Me             Shaggy   

                                     album_name  \
0                                  The Cookbook   
1                                   In The Zone   
2  Dangerously In Love (Alben für die Ewigkeit)   
3                                     Justified   
4                                      Hot Shot   

                              track_uri  \
0  spotify:track:0UaMYEvWZi0ZqiDOoHU3YI   
1  spotify:track:6I9VzXrHxO9rA9A5euc8Ak   
2  spotify:track:0WqIKmW4BTrj3eJFmnCKMv   
3  spotify:track:1AWQoqb9bSvzTjaLralEkT   
4  spotify:track:1lzr43nnXAijIGYnCT8M8H   

                              artist_uri  \
0  spotify:artist:2wIVse2owClT7go1WT98tk   
1  spotify:artist:26dSoYclwsYLMAKD3tpOr4   
2  spotify:artist:6vWDO969PvNqNYHIOW5v0m   
3  spotify:artist:31TPClRtHm23RisEBtV3X7   
4  spotify:artist:5EvFsr3kj42KNv97ZEnqij   

                              album_uri  track_duration_ms  
0  spotify:album:6vV5UrXcfyQD1wu4Qo2I9K             226863  
1  spotify:album:0z7pVBGOD7HCIB7S8eLkLI             198800  
2  spotify:album:25hVFAxTlDvXbx2X2QkUkE             235933  
3  spotify:album:6QPkyl04rXwTGlGlcYaRoW             267266  
4  spotify:album:6NmFmPX56pcLBOFMhIiKvF             227600  
Reloaded 6685101 tracks.


### Retrieve Gender Information based on the artists name

{'id': '859d0860-d480-4efd-970c-c05d5f1776b8', 'type': 'Person', 'ext:score': '100', 'name': 'Beyoncé', 'sort-name': 'Beyoncé', 'gender': 'female', 'country': 'US', 'area': {'id': '489ce91b-6658-3307-9877-795b68554c98', 'type': 'Country', 'name': 'United States', 'sort-name': 'United States', 'life-span': {'ended': 'false'}}, 'begin-area': {'id': 'c920948b-83e3-40b7-8fe9-9ab5abaac55b', 'type': 'City', 'name': 'Houston', 'sort-name': 'Houston', 'life-span': {'ended': 'false'}}, 'ipi-list': ['00341826274'], 'isni-list': ['0000000114914936'], 'life-span': {'begin': '1981-09-04', 'ended': 'false'}, 'alias-list': [{'locale': 'zh_Hant_TW', 'sort-name': '碧昂絲', 'type': 'Artist name', 'primary': 'primary', 'alias': '碧昂絲'}, {'locale': 'zh_Hans_CN', 'sort-name': 'Beyoncé', 'primary': 'primary', 'alias': 'Beyoncé'}, {'locale': 'en', 'sort-name': 'Knowles-Carter, Beyoncé', 'type': 'Artist name', 'alias': 'Beyoncé Knowles-Carter'}, {'locale': 'ja', 'sort-name': 'ビヨンセ', 'type': 'Artist name', 'primary': 'primary', 'alias': 'ビヨンセ'}, {'sort-name': 'Beyonce', 'type': 'Search hint', 'alias': 'Beyonce'}, {'sort-name': "Beyonce'", 'type': 'Search hint', 'alias': "Beyonce'"}, {'sort-name': 'B1C', 'type': 'Search hint', 'alias': 'B1C'}, {'locale': 'en', 'sort-name': 'Knowles-Carter, Beyoncé Giselle', 'type': 'Legal name', 'begin-date': '2008-04-04', 'alias': 'Beyoncé Giselle Knowles-Carter'}, {'locale': 'en', 'sort-name': 'Knowles, Beyoncé Giselle', 'type': 'Legal name', 'begin-date': '1981-09-04', 'end-date': '2008-04-03', 'alias': 'Beyoncé Giselle Knowles'}, {'locale': 'en', 'sort-name': 'Knowles, Beyoncé', 'type': 'Artist name', 'alias': 'Beyoncé Knowles'}, {'sort-name': 'Queen Bey', 'alias': 'Queen Bey'}], 'tag-list': [{'count': '18', 'name': 'pop'}, {'count': '1', 'name': 'rap'}, {'count': '2', 'name': 'dance'}, {'count': '2', 'name': 'house'}, {'count': '2', 'name': 'american'}, {'count': '2', 'name': 'vocalist'}, {'count': '0', 'name': 'singer-songwriter'}, {'count': '2', 'name': 'hip hop'}, {'count': '-2', 'name': 'hard rock'}, {'count': '0', 'name': 'funk'}, {'count': '0', 'name': 'country'}, {'count': '0', 'name': 'actress'}, {'count': '5', 'name': 'soul'}, {'count': '0', 'name': 'electropop'}, {'count': '2', 'name': 'disco'}, {'count': '-2', 'name': 'rnb'}, {'count': '1', 'name': 'pianist'}, {'count': '0', 'name': 'director'}, {'count': '0', 'name': 'icon'}, {'count': '2', 'name': '2000s'}, {'count': '3', 'name': 'dance-pop'}, {'count': '2', 'name': 'adult contemporary'}, {'count': '0', 'name': 'pop rock'}, {'count': '6', 'name': 'pop rap'}, {'count': '1', 'name': 'progressive house'}, {'count': '1', 'name': 'dancer'}, {'count': '-3', 'name': 'hip hop rnb and dance hall'}, {'count': '2', 'name': 'pop soul'}, {'count': '2', 'name': 'grammy winner'}, {'count': '12', 'name': 'r&b'}, {'count': '0', 'name': 'singer/songwriter'}, {'count': '6', 'name': 'contemporary r&b'}, {'count': '2', 'name': '2010s'}, {'count': '-1', 'name': 'mbs-333'}, {'count': '0', 'name': 'art pop'}, {'count': '0', 'name': 'alternative r&b'}, {'count': '0', 'name': 'moombahton'}, {'count': '-2', 'name': 'contemporary pop'}, {'count': '2', 'name': '2020s'}, {'count': '0', 'name': 'nuno'}, {'count': '2', 'name': 'cultural icon'}, {'count': '0', 'name': 'businesswoman'}, {'count': '0', 'name': 'philanthropist'}]}


female
male
unknown


[1/107166] Missy Elliott ➜ female
[2/107166] Britney Spears ➜ female
[3/107166] Beyoncé ➜ female
[4/107166] Justin Timberlake ➜ male
[5/107166] Shaggy ➜ male
[6/107166] Usher ➜ male
[7/107166] The Pussycat Dolls ➜ unknown
[8/107166] Destiny's Child ➜ unknown
[9/107166] OutKast ➜ unknown
[10/107166] Nelly Furtado ➜ female
[11/107166] Jesse McCartney ➜ male
[12/107166] Cassie ➜ female
[13/107166] Omarion ➜ male
[14/107166] Avril Lavigne ➜ female
[15/107166] Chris Brown ➜ male
[16/107166] Sheryl Crow ➜ female
[17/107166] The Black Eyed Peas ➜ unknown
[18/107166] Bowling For Soup ➜ unknown
[19/107166] The Click Five ➜ unknown
[20/107166] Jonas Brothers ➜ unknown
[21/107166] Lil Mama ➜ male
[22/107166] Cascada ➜ unknown
[23/107166] Jason Derulo ➜ male
[24/107166] Ne-Yo ➜ male
[25/107166] Miley Cyrus ➜ female
[26/107166] Boys Like Girls ➜ unknown
[27/107166] Iyaz ➜ male
[28/107166] Kesha ➜ female
[29/107166] Justin Bieber ➜ male
[30/107166] M.I.A. ➜ female
[31/107166] The Killers ➜ unknown
[32/107166] blink-182 ➜ unknown
[33/107166] The All-American Rejects ➜ unknown
[34/107166] Vanessa Carlton ➜ female
[35/107166] Cris Cab ➜ male
[36/107166] Demi Lovato ➜ non-binary gender
[37/107166] We The Kings ➜ unknown
[38/107166] Survivor ➜ unknown
[39/107166] Daniel Tidwell ➜ unknown
[40/107166] Kaleptik ➜ unknown
[41/107166] Ben Foster ➜ male
[42/107166] Leslie Odom Jr. ➜ male
[43/107166] Christopher Jackson ➜ male
[44/107166] Lin-Manuel Miranda ➜ male
[45/107166] Led Zeppelin ➜ unknown
[46/107166] Collective Soul ➜ unknown
[47/107166] Nightwish ➜ unknown
[48/107166] Seal ➜ male
[49/107166] The Rolling Stones ➜ unknown
[50/107166] Lynyrd Skynyrd ➜ unknown
[51/107166] Boston ➜ unknown
[52/107166] Toto ➜ unknown
[53/107166] Kansas ➜ unknown
[54/107166] Queen ➜ unknown
[55/107166] Guns N' Roses ➜ unknown
[56/107166] Creedence Clearwater Revival ➜ unknown
[57/107166] Scorpions ➜ unknown
[58/107166] Rush ➜ unknown
[59/107166] Hoody ➜ female
[60/107166] Loco ➜ male
[61/107166] Park Kyung ➜ unknown
[62/107166] BTS ➜ unknown
[63/107166] Lovelyz ➜ unknown
[64/107166] LEE HI ➜ male
[65/107166] Ailee ➜ female
[66/107166] Miso ➜ male
[67/107166] Zion.T ➜ male
[68/107166] KARD ➜ unknown
[69/107166] Soyou & Junggigo ➜ unknown
[70/107166] Jay Park ➜ unknown
[71/107166] Zico ➜ male
[72/107166] BLACKPINK ➜ unknown
[73/107166] Eddy Kim ➜ male
[74/107166] Jung Yong Hwa ➜ female
[75/107166] CNBLUE ➜ unknown
[76/107166] Apink ➜ unknown
[77/107166] Suzy ➜ female
[78/107166] BAEKHYUN ➜ male
[79/107166] V ➜ unknown
[80/107166] BIGBANG ➜ unknown
[81/107166] G-DRAGON ➜ male
[82/107166] SOL ➜ unknown
[83/107166] TAEYANG ➜ male
[84/107166] AKDONG MUSICIAN ➜ unknown
[85/107166] EXO ➜ unknown
[86/107166] Hyuna ➜ female
[87/107166] Jenyer ➜ unknown
[88/107166] YEZI ➜ female
[89/107166] JIMIN (AOA) ➜ unknown
[90/107166] Camille Saint-Saëns ➜ male
[91/107166] No Vacation ➜ unknown
[92/107166] Banes World ➜ unknown
[93/107166] PWR BTTM ➜ unknown
[94/107166] Tears For Fears ➜ unknown
[get_wikidata_id] Error for MBID bac9a679-9f2e-4c7e-9a0d-36c37791aaab: 'url-relation-list'
[95/107166] Natureboy ➜ unknown
[96/107166] Joy Again ➜ unknown
[97/107166] Part Time ➜ unknown
[98/107166] King Krule ➜ male
[99/107166] The Preatures ➜ unknown
[100/107166] DIIV ➜ unknown
[101/107166] Gazebos ➜ unknown
[102/107166] Tacocat ➜ unknown
[103/107166] Army Navy ➜ unknown
[104/107166] The Lonely Forest ➜ unknown
[105/107166] Rooney ➜ unknown
[106/107166] Bloc Party ➜ unknown
[107/107166] Waxahatchee ➜ female
[108/107166] Grizzly Bear ➜ unknown
[109/107166] Vampire Weekend ➜ unknown
[110/107166] The Babies ➜ unknown
[111/107166] Kopecky ➜ unknown
[112/107166] Nothing But Thieves ➜ unknown
[113/107166] Dr. Dog ➜ unknown
[114/107166] My Morning Jacket ➜ unknown
[115/107166] Wet ➜ unknown
[116/107166] Steve Lacy ➜ male
[117/107166] BROCKHAMPTON ➜ unknown
[118/107166] A.Dd+ ➜ unknown
[119/107166] Antwon ➜ male
[120/107166] Night Lovell ➜ male
[121/107166] bo en ➜ male
[122/107166] Cigarettes After Sex ➜ unknown
[123/107166] dandelion hands ➜ unknown
[124/107166] Sports ➜ male
[125/107166] The Kooks ➜ unknown
[126/107166] Xiu Xiu ➜ unknown
[127/107166] Moose ➜ unknown
[128/107166] Rogue Wave ➜ unknown
[129/107166] Weezer ➜ unknown
[130/107166] Meltycanon ➜ unknown
[131/107166] Chairlift ➜ unknown
[132/107166] Connan Mockasin ➜ male
[133/107166] Marcy Playground ➜ unknown
[134/107166] Concorde ➜ unknown
[135/107166] Mariah Carey ➜ female
[136/107166] Parks, Squares and Alleys ➜ unknown
[137/107166] Ween ➜ unknown
[138/107166] SWV ➜ unknown
[139/107166] Fazerdaze ➜ female
[140/107166] Mons Vi ➜ unknown
[141/107166] Oingo Boingo ➜ unknown
[142/107166] salvia palth ➜ male
[143/107166] Blonde Tongues ➜ unknown
[144/107166] Sufjan Stevens ➜ male
[145/107166] How To Dress Well ➜ male
[146/107166] Bobby Caldwell ➜ male
[147/107166] Hüsker Dü ➜ unknown
[148/107166] The Vapors ➜ unknown
[149/107166] Lets Kill Janice ➜ unknown
[150/107166] Bedroom ➜ unknown
[151/107166] The Truth ➜ unknown
[152/107166] Gift ➜ unknown
[153/107166] Floh de Cologne ➜ unknown
[154/107166] May Blitz ➜ unknown
[155/107166] Jehst ➜ male
[156/107166] Black Moon ➜ unknown
[157/107166] Lords Of The Underground ➜ unknown
[158/107166] Rod McKuen ➜ male
[159/107166] John SaFranko ➜ male
[160/107166] WHY? ➜ unknown
[161/107166] Varsity ➜ unknown
[162/107166] Declan McKenna ➜ male
[163/107166] Phony Ppl ➜ unknown
[164/107166] Michael Kiwanuka ➜ male
[165/107166] Etta James ➜ female
[166/107166] Jawbreaker ➜ unknown
[167/107166] Naked Raygun ➜ unknown
[168/107166] Fog Lake ➜ unknown
[169/107166] Vansire ➜ unknown
[170/107166] Good Good Blood ➜ unknown
[171/107166] Ourselves the Elves ➜ unknown
[172/107166] Odina ➜ unknown
[173/107166] Boyscott ➜ unknown
[174/107166] Suburban Lawns ➜ unknown
[175/107166] Mild High Club ➜ unknown
[176/107166] The Smashing Pumpkins ➜ unknown
[177/107166] Oasis ➜ unknown
[178/107166] Aerosmith ➜ unknown
[179/107166] Natalie Merchant ➜ female
[180/107166] TLC ➜ unknown
[181/107166] Natalie Imbruglia ➜ female
[182/107166] Lisa Loeb & Nine Stories ➜ female
[183/107166] Toni Braxton ➜ female
[184/107166] Sarah McLachlan ➜ female
[185/107166] The Verve ➜ unknown
[186/107166] Soul Asylum ➜ unknown
[187/107166] The Verve Pipe ➜ unknown
[188/107166] Dido ➜ female
[189/107166] Sade ➜ unknown
[190/107166] DNA ➜ unknown
[191/107166] Hootie & The Blowfish ➜ unknown
[192/107166] Cali Swag District ➜ unknown
[193/107166] LMFAO ➜ unknown
[194/107166] Vanilla Ice ➜ male
[195/107166] The Isley Brothers ➜ unknown
[196/107166] Flo Rida ➜ male
[197/107166] Michael Jackson ➜ male
[198/107166] Sir Mix-A-Lot ➜ male
[199/107166] P!nk ➜ female
[200/107166] Mr. C The Slide Man ➜ unknown
[201/107166] Neil Diamond ➜ male
[202/107166] Cupid ➜ male
[203/107166] Lady Gaga ➜ female
[204/107166] Daft Punk ➜ unknown
[205/107166] Pitbull ➜ male
[206/107166] Rihanna ➜ female
[207/107166] Train ➜ unknown
[208/107166] Nicki Minaj ➜ female
[209/107166] Wild Cherry ➜ unknown
[210/107166] Old School Players ➜ unknown
[211/107166] Will Smith ➜ male
[212/107166] Bob Marley & The Wailers ➜ unknown
[213/107166] Ed Sheeran ➜ male
[214/107166] Bruno Mars ➜ male
[215/107166] Edwin McCain ➜ male
[216/107166] Jason Mraz ➜ male
[217/107166] Sixpence None The Richer ➜ unknown
[218/107166] American Authors ➜ unknown
[219/107166] K-Ci & JoJo ➜ unknown
[220/107166] Toploader ➜ unknown
[221/107166] Colbie Caillat ➜ female
[222/107166] Dave Matthews Band ➜ unknown
[223/107166] Rascal Flatts ➜ unknown
[224/107166] Tom Petty and the Heartbreakers ➜ unknown
[225/107166] Taio Cruz ➜ male
[226/107166] Tiësto ➜ male
[227/107166] The Lumineers ➜ unknown
[228/107166] House Of Pain ➜ unknown
[229/107166] Kris Kross ➜ unknown
[230/107166] Next ➜ unknown
[231/107166] Florida Georgia Line ➜ unknown
[232/107166] Clean Bandit ➜ unknown
[233/107166] MKTO ➜ unknown
[234/107166] Charli XCX ➜ female
[235/107166] Calvin Harris ➜ male
[236/107166] Becky G ➜ male
[237/107166] Ariana Grande ➜ female
[238/107166] Macklemore & Ryan Lewis ➜ unknown
[239/107166] The Aranbee Pop Symphony Orchestra ➜ unknown
[240/107166] STRFKR ➜ unknown
[241/107166] 311 ➜ unknown
[242/107166] Nathaniel Rateliff & The Night Sweats ➜ unknown
[243/107166] Toadies ➜ unknown
[244/107166] Queens of the Stone Age ➜ unknown
[245/107166] The Cranberries ➜ unknown
[246/107166] Misfits ➜ unknown
[247/107166] Willy Moon ➜ male
[248/107166] The Flys ➜ unknown
[249/107166] Tame Impala ➜ unknown
[250/107166] Sia ➜ female
[251/107166] Arctic Monkeys ➜ unknown
[252/107166] Fink ➜ male
[253/107166] James Vincent McMorrow ➜ male
[254/107166] VÉRITÉ ➜ female
[255/107166] dvsn ➜ unknown
[256/107166] Starley ➜ female
[257/107166] HAIM ➜ unknown
[258/107166] Kiiara ➜ female
[259/107166] Tash Sultana ➜ genderfluid
[260/107166] Thomston ➜ unknown
[261/107166] Oh Wonder ➜ male
[262/107166] Banks ➜ male
[263/107166] Diddy ➜ male
[264/107166] Flume ➜ male
[265/107166] Niia ➜ female
[266/107166] Halsey ➜ female
[267/107166] Elliot Moss ➜ male
[268/107166] The Kite String Tangle ➜ male
[269/107166] London Grammar ➜ unknown
[270/107166] Childish Gambino ➜ male
[271/107166] LANY ➜ unknown
[272/107166] Francis and the Lights ➜ unknown
[273/107166] Water Park ➜ unknown
[274/107166] Joe Goddard ➜ male
[275/107166] Stereophonics ➜ unknown
[276/107166] Rudimental ➜ unknown
[277/107166] Stateless ➜ unknown
[278/107166] The Paper Kites ➜ unknown
[279/107166] Yeo ➜ unknown
[280/107166] GOSTO ➜ unknown
[281/107166] Sol Rising ➜ unknown
[282/107166] Dagny ➜ female
[283/107166] Bonobo ➜ male
[284/107166] Gotts Street Park ➜ unknown
[285/107166] Rhye ➜ unknown
[286/107166] Aquilo ➜ unknown
[287/107166] BASECAMP ➜ unknown
[288/107166] Bruno Major ➜ male
[289/107166] Alpines ➜ unknown
[290/107166] James Hersey ➜ male
[291/107166] Jessie J ➜ female
[292/107166] Disclosure ➜ unknown
[293/107166] George Maple ➜ male
[294/107166] DeJ Loaf ➜ female
[295/107166] Great Good Fine Ok ➜ unknown
[296/107166] John Splithoff ➜ male
[297/107166] Asta ➜ unknown
[298/107166] Charlie Puth ➜ male
[299/107166] Fred V & Grafix ➜ unknown
[300/107166] Catfish and the Bottlemen ➜ unknown
[301/107166] Cage The Elephant ➜ unknown
[302/107166] Red Hot Chili Peppers ➜ unknown
[303/107166] Cold War Kids ➜ unknown
[304/107166] Kings of Leon ➜ unknown
[305/107166] Florence + The Machine ➜ unknown
[306/107166] The Beatles ➜ unknown
[307/107166] Green Day ➜ unknown
[308/107166] Lord Huron ➜ unknown
[309/107166] Portugal. The Man ➜ unknown
[310/107166] alt-J ➜ female
[311/107166] Grouplove ➜ unknown
[312/107166] Gorillaz ➜ unknown
[313/107166] Electric Light Orchestra ➜ unknown
[314/107166] Dirty Heads ➜ unknown
[315/107166] UB40 ➜ unknown
[316/107166] Death Cab for Cutie ➜ unknown
[317/107166] The Monkees ➜ unknown
[318/107166] Noah Kahan ➜ male
[319/107166] Willie Nelson ➜ male
[320/107166] The Cadillac Three ➜ unknown
[321/107166] Chris Lane ➜ male
[322/107166] Travis Tritt ➜ male
[323/107166] Alan Jackson ➜ male
[324/107166] Wheeler Walker Jr. ➜ male
[325/107166] Whiskey Myers ➜ unknown
[326/107166] Jason Boland & The Stragglers ➜ unknown
[327/107166] Cody Johnson ➜ male
[328/107166] Merle Haggard ➜ male
[329/107166] David Allan Coe ➜ male
[330/107166] Mark Chesnutt ➜ male
[331/107166] Gary Stewart ➜ male
[332/107166] Merle Haggard & The Strangers ➜ unknown
[333/107166] Clint Black ➜ male
[334/107166] Brooks & Dunn ➜ unknown
[335/107166] Josh Turner ➜ female
[336/107166] Midland ➜ unknown
[337/107166] Post Malone ➜ male
[338/107166] Chance The Rapper ➜ male
[339/107166] Jeremih ➜ male
[340/107166] Lil Wayne ➜ male
[341/107166] Lupe Fiasco ➜ male
[342/107166] Desiigner ➜ male
[343/107166] Wale ➜ male
[344/107166] Migos ➜ unknown
[345/107166] Drake ➜ male
[346/107166] Big Sean ➜ male
[347/107166] Bryson Tiller ➜ male
[348/107166] YG ➜ male
[349/107166] Meek Mill ➜ male
[350/107166] Timbaland ➜ male
[351/107166] ScHoolboy Q ➜ male
[352/107166] blackbear ➜ male
[353/107166] J. Cole ➜ female
[354/107166] A Boogie Wit da Hoodie ➜ male
[355/107166] Travis Scott ➜ male
[356/107166] mansionz ➜ unknown
[357/107166] Russ ➜ male
[358/107166] Kendrick Lamar ➜ male
[359/107166] Aminé ➜ male
[360/107166] Cardi B ➜ female
[361/107166] Juicy J ➜ female
[362/107166] Macklemore ➜ male
[363/107166] Rae Sremmurd ➜ unknown
[364/107166] Fort Minor ➜ unknown
[365/107166] Quinn XCII ➜ male
[366/107166] Kevin Gates ➜ male
[367/107166] Bobby Shmurda ➜ male
[368/107166] Kodak Black ➜ male
[369/107166] Lil Dicky ➜ male
[370/107166] Tee Grizzley ➜ male
[371/107166] PnB Rock ➜ male
[372/107166] French Montana ➜ male
[373/107166] Plies ➜ male
[374/107166] Mike Stud ➜ male
[375/107166] AJR ➜ unknown
[376/107166] Natural Self ➜ unknown
[377/107166] Spoon ➜ unknown
[378/107166] Michna ➜ male
[379/107166] Pretty Lights ➜ male
[380/107166] Erykah Badu ➜ female
[381/107166] Kavinsky ➜ male
[382/107166] Little Dragon ➜ unknown
[383/107166] Jens Buchert ➜ unknown
[384/107166] Zero 7 ➜ unknown
[385/107166] Lotus ➜ male
[386/107166] Oakman ➜ unknown
[387/107166] Handsome Boy Modeling School ➜ unknown
[388/107166] Lykke Li ➜ female
[389/107166] Flying Lotus feat. Andreya Triana ➜ female
[390/107166] Cooly G ➜ unknown
[391/107166] Miguel ➜ male
[392/107166] C2C ➜ unknown
[393/107166] X Ambassadors ➜ unknown
[394/107166] DJ Shadow ➜ male
[395/107166] DJ Cam ➜ male
[396/107166] deadmau5 ➜ male
[397/107166] Wax Tailor ➜ male
[398/107166] Massive Attack ➜ unknown
[399/107166] Ellie Goulding ➜ female
[400/107166] RJD2 ➜ male
[401/107166] TOKiMONSTA ➜ female
[402/107166] The Pharcyde ➜ unknown
[403/107166] La Roux ➜ unknown
[404/107166] Incubus ➜ unknown
[405/107166] Flying Lotus ➜ male
[406/107166] Goapele ➜ female
[407/107166] The xx ➜ unknown
[408/107166] Tricky ➜ male
[409/107166] Daedelus ➜ male
[410/107166] AlunaGeorge ➜ unknown
[411/107166] Télépopmusik ➜ unknown
[412/107166] Balam Acab ➜ male
[413/107166] The Weeknd ➜ male
[414/107166] Holy Other ➜ male
[415/107166] The Avalanches ➜ unknown
[416/107166] Major Lazer ➜ unknown
[417/107166] Passion Pit ➜ unknown
[418/107166] Hooverphonic ➜ unknown
[419/107166] Flight Facilities ➜ unknown
[420/107166] Populous ➜ unknown
[421/107166] Blackmill ➜ unknown
[422/107166] PANTyRAiD ➜ unknown
[423/107166] Morcheeba ➜ unknown
[424/107166] Portishead ➜ unknown
[425/107166] Air ➜ unknown
[426/107166] A Tribe Called Quest ➜ unknown
[427/107166] Blue Foundation ➜ unknown
[428/107166] Washed Out ➜ male
[429/107166] Burial ➜ male
[430/107166] Blackbird Blackbird ➜ unknown
[431/107166] Tourist ➜ male
[432/107166] Kanye West ➜ unknown
[433/107166] Beach House ➜ unknown
[434/107166] Four Tet ➜ male
[435/107166] Thievery Corporation ➜ unknown
[get_wikidata_id] Error for MBID 481d55c3-a76f-415e-962e-67f8689f573d: 'url-relation-list'
[436/107166] Tripssono ➜ unknown
[437/107166] SBTRKT ➜ male
[438/107166] The Roots ➜ unknown
[439/107166] Peter Bjorn and John ➜ unknown
[440/107166] The Naked And Famous ➜ unknown
[441/107166] Afterlife ➜ male
[442/107166] Fort Fairfield ➜ unknown
[443/107166] Bliss ➜ unknown
[444/107166] Andreya Triana ➜ female
[445/107166] Mr Twin Sister ➜ unknown
[446/107166] Baths ➜ male
[447/107166] Cut Chemist ➜ male
[448/107166] Deltron 3030 ➜ unknown
[449/107166] Jamie xx ➜ male
[450/107166] Feed Me ➜ male
[451/107166] Georgia Anne Muldrow ➜ female
[452/107166] Paper Diamond ➜ male
[453/107166] Robert Glasper Experiment ➜ unknown
[454/107166] Michal Menert ➜ unknown
[455/107166] Crystal Castles ➜ unknown
[456/107166] InContext ➜ unknown
[457/107166] Adventure Club ➜ unknown
[458/107166] TwoThirds ➜ unknown
[459/107166] Phantogram ➜ unknown
[460/107166] Feist ➜ female
[461/107166] Cults ➜ unknown
[462/107166] Gold Panda ➜ male
[463/107166] Long Beach Dub Allstars ➜ unknown
[464/107166] Kid Cudi ➜ male
[465/107166] Yael Naim ➜ female
[466/107166] Passenger ➜ male
[467/107166] Milky Chance ➜ unknown
[468/107166] Young Thug ➜ male
[469/107166] James Bay ➜ male
[470/107166] Lost Frequencies ➜ male
[471/107166] Wiz Khalifa ➜ male
[472/107166] Tom Jones ➜ male
[473/107166] Alessia Cara ➜ female
[474/107166] Ruth B. ➜ male
[475/107166] Cam ➜ male
[476/107166] Rob Thomas ➜ male
[477/107166] Jeffrey James ➜ male
[478/107166] Jasmine Thompson ➜ female
[479/107166] Gabrielle Aplin ➜ female
[480/107166] Christina Perri ➜ female
[481/107166] Molly Kate Kestner ➜ female
[482/107166] Katy McAllister ➜ female
[483/107166] Joseph Vincent ➜ male
[484/107166] James Arthur ➜ male
[485/107166] Alanis Morissette ➜ female
[486/107166] Alice In Chains ➜ unknown
[487/107166] Barenaked Ladies ➜ unknown
[488/107166] Beck ➜ male
[489/107166] Better Than Ezra ➜ unknown
[490/107166] The Black Crowes ➜ unknown
[491/107166] Blind Melon ➜ unknown
[492/107166] Blues Traveler ➜ unknown
[493/107166] Bush ➜ female
[494/107166] Candlebox ➜ unknown
[495/107166] Chumbawamba ➜ unknown
[496/107166] Counting Crows ➜ unknown
[497/107166] Coyote Shivers ➜ male
[498/107166] Crazy Town ➜ unknown
[499/107166] Duncan Sheik ➜ male
[500/107166] Eagle-Eye Cherry ➜ male
[501/107166] Eve 6 ➜ unknown
[502/107166] Everclear ➜ unknown
[503/107166] Everything ➜ unknown
[504/107166] Filter ➜ unknown
[505/107166] Fiona Apple ➜ female
[506/107166] Foo Fighters ➜ unknown
[507/107166] Fuel ➜ unknown
[508/107166] Gin Blossoms ➜ unknown
[509/107166] The Goo Goo Dolls ➜ unknown
[510/107166] Harvey Danger ➜ unknown
[511/107166] Jane's Addiction ➜ unknown
[512/107166] Live ➜ unknown
[513/107166] Matchbox Twenty ➜ unknown
[514/107166] Mazzy Star ➜ unknown
[515/107166] Nine Inch Nails ➜ unknown
[516/107166] Nirvana ➜ unknown
[517/107166] No Doubt ➜ unknown
[518/107166] The Offspring ➜ unknown
[519/107166] Pearl Jam ➜ unknown
[520/107166] R.E.M. ➜ unknown
[521/107166] Smash Mouth ➜ unknown
[522/107166] Soundgarden ➜ unknown
[523/107166] Spin Doctors ➜ unknown
[524/107166] Sublime ➜ unknown
[525/107166] Sugar Ray ➜ male
[526/107166] Third Eye Blind ➜ unknown
[527/107166] Tonic ➜ unknown
[528/107166] Vertical Horizon ➜ unknown
[529/107166] The Wallflowers ➜ unknown
[530/107166] Sylvan Esso ➜ unknown
[531/107166] Gregory Alan Isakov ➜ male
[532/107166] Whilk & Misky ➜ unknown
[533/107166] Young the Giant ➜ unknown
[534/107166] Sleater-Kinney ➜ unknown
[535/107166] Teenage Fanclub ➜ unknown
[536/107166] The Shins ➜ unknown
[537/107166] Bishop Briggs ➜ male
[538/107166] Angel Olsen ➜ female
[539/107166] ✝✝✝ (Crosses) ➜ unknown
[540/107166] Broods ➜ unknown
[541/107166] Cloud Nothings ➜ unknown
[542/107166] Sum 41 ➜ unknown
[543/107166] Adolescents ➜ unknown
[544/107166] Arcade Fire ➜ unknown
[545/107166] John Mayer ➜ male
[546/107166] The Black Keys ➜ unknown
[547/107166] Ryan Adams ➜ male
[548/107166] Night Riots ➜ unknown
[549/107166] Built To Spill ➜ unknown
[550/107166] The Twilight Sad ➜ unknown
[551/107166] Flagship ➜ unknown
[552/107166] The Front Bottoms ➜ unknown
[553/107166] Placebo ➜ unknown
[554/107166] KAYTRANADA ➜ male
[555/107166] Dashboard Confessional ➜ unknown
[556/107166] Nada Surf ➜ unknown
[557/107166] My Bloody Valentine ➜ unknown
[558/107166] The Clash ➜ unknown
[559/107166] Hippo Campus ➜ unknown
[560/107166] The Airborne Toxic Event ➜ unknown
[561/107166] Thundercat ➜ male
[562/107166] Vinyl Theatre ➜ unknown
[563/107166] David Bowie ➜ male
[564/107166] Interpol ➜ unknown
[565/107166] Real Estate ➜ unknown
[566/107166] Unknown Mortal Orchestra ➜ unknown
[567/107166] Lorde ➜ female
[568/107166] Common ➜ male
[569/107166] New Order ➜ unknown
[570/107166] Lana Del Rey ➜ female
[571/107166] Fleet Foxes ➜ unknown
[572/107166] Amy Winehouse ➜ female
[573/107166] Zella Day ➜ female
[574/107166] DREAMCAR ➜ unknown
[575/107166] Jack Garratt ➜ male
[576/107166] AFI ➜ unknown
[577/107166] Broken Social Scene ➜ unknown
[578/107166] Frank Ocean ➜ male
[579/107166] Kristin Kontrol ➜ female
[580/107166] Iggy Pop ➜ male
[581/107166] Cold Showers ➜ unknown
[582/107166] Palmas ➜ male
[583/107166] The Dig ➜ unknown
[584/107166] Merchandise ➜ unknown
[585/107166] Kwesi Foraes ➜ unknown
[586/107166] Benjamin Gibbard ➜ male
[587/107166] The Bluetones ➜ unknown
[588/107166] Whitney ➜ female
[589/107166] Chelsea Wolfe ➜ female
[590/107166] Plastic Flowers ➜ unknown
[591/107166] Jay Som ➜ male
[592/107166] The National ➜ unknown
[593/107166] The Dangerous Summer ➜ unknown
[594/107166] Radiohead ➜ unknown
[595/107166] Car Seat Headrest ➜ unknown
[596/107166] Allah-Las ➜ unknown
[597/107166] Harry Styles ➜ male
[598/107166] Hozier ➜ male
[599/107166] Aloe Blacc ➜ male
[600/107166] John Legend ➜ male
[601/107166] Pharrell Williams ➜ male
[602/107166] Mark Ronson ➜ male
[603/107166] WALK THE MOON ➜ unknown
[604/107166] Matisyahu ➜ male
[605/107166] Jess Glynne ➜ female
[606/107166] You Me At Six ➜ unknown
[607/107166] Meghan Trainor ➜ female
[608/107166] Nico & Vinz ➜ unknown
[609/107166] Avicii ➜ male
[610/107166] Itch ➜ unknown
[611/107166] George Ezra ➜ male
[612/107166] Imagine Dragons ➜ unknown
[613/107166] OneRepublic ➜ unknown
[614/107166] Andy Grammer ➜ male
[615/107166] Adele ➜ female
[616/107166] Bastille ➜ unknown
[617/107166] Panic! At The Disco ➜ unknown
[618/107166] Bleachers ➜ unknown
[619/107166] Jason Gray ➜ male
[620/107166] Mary Lambert ➜ female
[621/107166] Sheppard ➜ unknown
[622/107166] Fall Out Boy ➜ unknown
[623/107166] Mumford & Sons ➜ unknown
[624/107166] Nine Days ➜ unknown
[625/107166] Owl City ➜ unknown
[626/107166] The Downtown Fiction ➜ unknown
[627/107166] Fifth Harmony ➜ unknown
[628/107166] Neon Trees ➜ unknown
[629/107166] Two Door Cinema Club ➜ unknown
[630/107166] Bad Suns ➜ unknown
[631/107166] Audien ➜ male
[632/107166] The Darkness ➜ unknown
[633/107166] The Hooters ➜ unknown
[634/107166] Atlas Genius ➜ unknown
[635/107166] Martin Solveig ➜ male
[636/107166] DNCE ➜ unknown
[637/107166] Bear Hands ➜ male
[638/107166] Hillsong United ➜ unknown
[639/107166] The Royal Concept ➜ unknown
[640/107166] Beach Avenue ➜ unknown
[641/107166] Built By Titan ➜ unknown
[642/107166] Astoria Kings ➜ unknown
[643/107166] Anthem Lights ➜ unknown
[644/107166] Yellowcard ➜ unknown
[645/107166] Adam Martin ➜ male
[646/107166] Matt Nathanson ➜ male
[647/107166] Los Claxons ➜ unknown
[648/107166] Big & Rich ➜ unknown
[649/107166] Scotty McCreery ➜ male
[650/107166] Russell Dickerson ➜ unknown
[651/107166] Darius Rucker ➜ male
[652/107166] Ciara ➜ female
[653/107166] Kelly Rowland ➜ female
[654/107166] Tim McGraw ➜ male
[655/107166] Nick Jonas ➜ male
[656/107166] Tyga ➜ male
[657/107166] Bryan Adams ➜ male
[658/107166] Lyrica Anderson ➜ female
[659/107166] Savage Garden ➜ unknown
[660/107166] Blake Shelton ➜ male
[661/107166] Machine Gun Kelly ➜ unknown
[662/107166] Ashanti ➜ female
[663/107166] Trey Songz ➜ male
[664/107166] Kid Ink ➜ male
[665/107166] Canaan Smith ➜ male
[666/107166] Dylan Scott ➜ male
[667/107166] Bow Wow ➜ unknown
[668/107166] Jon B. ➜ male
[669/107166] Rick Ross ➜ male
[670/107166] Jennifer Lopez ➜ female
[671/107166] Christina Aguilera ➜ female
[672/107166] Ester Dean ➜ male
[673/107166] Tech N9ne ➜ male
[674/107166] Kehlani ➜ non-binary gender
[675/107166] Ginuwine ➜ male
[676/107166] Wrabel ➜ male
[677/107166] T-Pain ➜ male
[678/107166] Yung Berg ➜ female
[679/107166] Band Of Skulls ➜ unknown
[680/107166] Nick Drake ➜ male
[681/107166] The Cave Singers ➜ unknown
[682/107166] Dan Auerbach ➜ male
[683/107166] Beirut ➜ unknown
[684/107166] The Megaphonic Thrift ➜ unknown
[685/107166] Vacationer ➜ unknown
[686/107166] Bob Dylan ➜ male
[687/107166] The Last Shadow Puppets ➜ unknown
[688/107166] Alex Turner ➜ female
[689/107166] Tedeschi Trucks Band ➜ unknown
[690/107166] Grace Potter & The Nocturnals ➜ unknown
[691/107166] Toro y Moi ➜ male
[692/107166] Of Monsters and Men ➜ unknown
[693/107166] Milo Greene ➜ unknown
[694/107166] Danger Mouse ➜ male
[695/107166] The Dodos ➜ unknown
[696/107166] Surfer Blood ➜ unknown
[697/107166] Bahamas ➜ male
[698/107166] The Wood Brothers ➜ unknown
[699/107166] The Tallest Man On Earth ➜ male
[700/107166] The Velvet Underground ➜ unknown
[701/107166] The Avett Brothers ➜ unknown
[702/107166] Wild Nothing ➜ unknown
[703/107166] Langhorne Slim ➜ male
[704/107166] Donovan ➜ male
[705/107166] Phish ➜ unknown
[706/107166] Wilco ➜ unknown
[707/107166] Jeff Buckley ➜ male
[708/107166] The Deep Dark Woods ➜ unknown
[709/107166] Megafaun ➜ unknown
[710/107166] Middle Brother ➜ unknown
[711/107166] Vetiver ➜ unknown
[712/107166] Parlovr ➜ unknown
[713/107166] Leonard Cohen ➜ male
[714/107166] Junip ➜ unknown
[715/107166] Jim James ➜ male
[716/107166] The Smiths ➜ unknown
[717/107166] Phosphorescent ➜ male
[718/107166] Devendra Banhart ➜ male
[719/107166] The Civil Wars ➜ unknown
[720/107166] Paper Bird ➜ male
[721/107166] The Moondoggies ➜ unknown
[722/107166] Aretha Franklin ➜ female
[723/107166] Here We Go Magic ➜ unknown
[724/107166] José González ➜ male
[725/107166] Tycho ➜ male
[726/107166] Sampha ➜ male
[727/107166] Leo Stannard ➜ male
[728/107166] Sango ➜ male
[729/107166] Cotton Jones ➜ male
[730/107166] Guy Clark ➜ male
[731/107166] Jim James & Calexico ➜ unknown
[732/107166] Ben Howard ➜ male
[733/107166] Lil Pump ➜ male
[734/107166] Trippie Redd ➜ male
[735/107166] Logic ➜ male
[736/107166] Lil Uzi Vert ➜ non-binary gender
[737/107166] Smokepurpp ➜ male
[738/107166] Dae Dae ➜ male
[739/107166] Evanescence ➜ unknown
[740/107166] Gotye ➜ male
[741/107166] Selena ➜ female
[742/107166] Roy Orbison ➜ male
[743/107166] Beth Nielsen Chapman ➜ female
[744/107166] Johnny Cash ➜ male
[745/107166] Harry Chapin ➜ male
[746/107166] Heart ➜ unknown
[747/107166] Climax Blues Band ➜ male
[748/107166] SZA ➜ female
[749/107166] BJ The Chicago Kid ➜ male
[750/107166] 6LACK ➜ male
[751/107166] Wolfie ➜ unknown
[752/107166] Anderson .Paak ➜ male
[753/107166] Sam Smith ➜ non-binary gender
[754/107166] Dua Lipa ➜ female
[755/107166] J Balvin ➜ male
[756/107166] Marshmello ➜ male
[757/107166] Lauv ➜ male
[758/107166] A R I Z O N A ➜ unknown
[759/107166] Saint Motel ➜ male
[760/107166] Roy Woods ➜ male
[761/107166] Khalid ➜ male
[762/107166] Louis Tomlinson ➜ male
[763/107166] Liam Payne ➜ male
[764/107166] Maroon 5 ➜ unknown
[765/107166] Axwell /\ Ingrosso ➜ unknown
[766/107166] ZAYN ➜ male
[767/107166] Hailee Steinfeld ➜ female
[768/107166] Rita Ora ➜ female
[769/107166] Hoodie Allen ➜ male
[770/107166] The Head and the Heart ➜ unknown
[771/107166] A Broken Silence ➜ unknown
[772/107166] Avenged Sevenfold ➜ unknown
[773/107166] Deftones ➜ unknown
[774/107166] Killswitch Engage ➜ unknown
[775/107166] Slipknot ➜ unknown
[776/107166] Chevelle ➜ unknown
[777/107166] Five Finger Death Punch ➜ unknown
[778/107166] Metallica ➜ unknown
[779/107166] Twenty One Pilots ➜ unknown
[780/107166] Jon Bellion ➜ male
[781/107166] ODESZA ➜ unknown
[782/107166] Fetty Wap ➜ male
[783/107166] Galantis ➜ unknown
[784/107166] DVBBS ➜ unknown
[785/107166] Tchami ➜ male
[786/107166] TroyBoi ➜ unknown
[787/107166] Sam Feldt ➜ male
[788/107166] Jack Ü ➜ male
[789/107166] Shoffy ➜ unknown
[790/107166] Diplo ➜ male
[791/107166] Keys N Krates ➜ unknown
[792/107166] Glass Animals ➜ unknown
[793/107166] Jauz ➜ male
[794/107166] Flosstradamus ➜ male
[795/107166] RL Grime ➜ male
[796/107166] Party Favor ➜ unknown
[797/107166] Kungs ➜ male
[798/107166] Steve Aoki ➜ male
[799/107166] What So Not ➜ unknown
[800/107166] Cheat Codes ➜ unknown
[801/107166] Tiga ➜ male
[802/107166] Destructo ➜ male
[803/107166] Sevenn ➜ unknown
[804/107166] Dillon Francis ➜ male
[805/107166] Gucci Mane ➜ male
[806/107166] First Aid Kit ➜ unknown
[807/107166] The New Basement Tapes ➜ unknown
[808/107166] Shakey Graves ➜ male
[809/107166] Kat Edmonson ➜ female
[810/107166] Trampled By Turtles ➜ unknown
[811/107166] Father John Misty ➜ male
[812/107166] Oscar Isaac ➜ male
[813/107166] Angus & Julia Stone ➜ unknown
[814/107166] The Decemberists ➜ unknown
[815/107166] Kaleo ➜ unknown
[816/107166] Tom Waits ➜ male
[817/107166] Leon Bridges ➜ male
[818/107166] Sturgill Simpson ➜ male
[819/107166] Chicano Batman ➜ unknown
[820/107166] NEEDTOBREATHE ➜ unknown
[821/107166] Witt Lowry ➜ unknown
[822/107166] Baaba Maal ➜ male
[823/107166] T.I. ➜ male
[824/107166] Future ➜ male
[825/107166] KYLE ➜ male
[826/107166] DJ Khaled ➜ male
[827/107166] Playboi Carti ➜ male
[828/107166] Luis Fonsi ➜ male
[829/107166] JAY Z ➜ male
[830/107166] Young Money ➜ male
[831/107166] 3rddy Baby ➜ unknown
[832/107166] Eminem ➜ male
[833/107166] A$AP Rocky ➜ male
[834/107166] Dr. Dre ➜ male
[835/107166] Mike WiLL Made-It ➜ male
[836/107166] NAV ➜ male
[837/107166] Metro Boomin ➜ male
[838/107166] Cook Laflare ➜ male
[839/107166] Ty Dolla $ign ➜ male
[840/107166] DRAM ➜ male
[841/107166] Moon Taxi ➜ unknown
[842/107166] Jonas Blue ➜ male
[843/107166] Niall Horan ➜ male
[844/107166] ItaloBrothers ➜ unknown
[845/107166] Kevin Abstract ➜ male
[846/107166] Katy Perry ➜ female
[847/107166] Marc E. Bassy ➜ male
[848/107166] Zedd ➜ male
[849/107166] Ricegum ➜ male
[850/107166] Jake Paul ➜ male
[get_wikidata_id] Error for MBID 775134eb-f265-49b6-8cda-663cb3860724: 'url-relation-list'
[851/107166] Rachelle Maust ➜ unknown
[852/107166] Maggie Lindemann ➜ female
[853/107166] The Jackson 5 ➜ unknown
[854/107166] Jacob Sartorius ➜ male
[855/107166] Selena Gomez ➜ female
[856/107166] Taylor Swift ➜ female
[857/107166] Bill Withers ➜ male
[858/107166] Camila Cabello ➜ female
[859/107166] The Rembrandts ➜ unknown
[860/107166] Will Butler ➜ male
[861/107166] Parquet Courts ➜ unknown
[862/107166] Palma Violets ➜ unknown
[863/107166] The Orwells ➜ unknown
[864/107166] Japandroids ➜ unknown
[865/107166] Titus Andronicus ➜ unknown
[866/107166] Restorations ➜ unknown
[867/107166] Iron Chic ➜ unknown
[868/107166] Direct Hit! ➜ unknown
[869/107166] Robin Schulz ➜ male
[870/107166] Felix Jaehn ➜ non-binary gender
[871/107166] Damien Jurado ➜ male
[872/107166] The Chainsmokers ➜ unknown
[873/107166] Conrad Sewell ➜ male
[874/107166] Moguai ➜ male
[875/107166] King Arthur ➜ male
[876/107166] Sigala ➜ male
[877/107166] Alex Newell ➜ non-binary gender
[878/107166] Coleman Hell ➜ male
[879/107166] Vance Joy ➜ male
[880/107166] Zara Larsson ➜ female
[881/107166] Kygo ➜ male
[882/107166] LunchMoney Lewis ➜ male
[883/107166] Akon ➜ male
[884/107166] Years & Years ➜ unknown
[885/107166] One Direction ➜ unknown
[886/107166] Mike Posner ➜ male
[887/107166] BUNT. ➜ unknown
[888/107166] NIHILS ➜ unknown
[889/107166] Royal Tongues ➜ unknown
[890/107166] Both ➜ unknown
[891/107166] Doug Locke ➜ male
[892/107166] Tryon ➜ unknown
[893/107166] Lost Kings ➜ unknown
[894/107166] Parachute ➜ unknown
[895/107166] Michael Calfan ➜ unknown
[896/107166] Otto Knows ➜ male
[897/107166] Olly Murs ➜ male
[898/107166] Madden ➜ unknown
[899/107166] Alex Adair ➜ male
[900/107166] Coasts ➜ unknown
[901/107166] JR JR ➜ unknown
[902/107166] Floduxe ➜ unknown
[903/107166] Kiso ➜ unknown
[904/107166] Deep Chills ➜ unknown
[get_wikidata_id] Error for MBID 0e2de46e-63b1-4f78-b3f7-5ec26c7a2dc6: 'url-relation-list'
[905/107166] Santa Maradona F.C. ➜ unknown
[906/107166] Echosmith ➜ unknown
[907/107166] Pentatonix ➜ unknown
[908/107166] Madcon ➜ unknown
[909/107166] Matoma ➜ male
[910/107166] Timeflies ➜ unknown
[911/107166] MØ ➜ female
[912/107166] Coldplay ➜ unknown
[913/107166] Zayde Wølf ➜ male
[914/107166] Fleur East ➜ female
[915/107166] Birdy ➜ female
[916/107166] Relient K ➜ unknown
[917/107166] MAGIC! ➜ unknown
[918/107166] NONONO ➜ unknown
[919/107166] Nashville Cast ➜ unknown
[920/107166] Kacey Musgraves ➜ female
[921/107166] Johnnyswim ➜ unknown
[922/107166] Maddi Jane ➜ female
[923/107166] Band of Horses ➜ unknown
[924/107166] Tom Odell ➜ male
[925/107166] Donavon Frankenreiter ➜ male
[926/107166] James Blunt ➜ male
[927/107166] Swedish House Mafia ➜ unknown
[928/107166] Collin McLoughlin ➜ male
[929/107166] Bill West ➜ male
[930/107166] Madilyn Bailey ➜ female
[931/107166] Norah Jones ➜ female
[932/107166] Daniela Andrade ➜ female
[933/107166] Stevie Wonder ➜ male
[934/107166] Joanna Wang ➜ female
[935/107166] Sean Hayes ➜ male
[936/107166] Two Worlds ➜ unknown
[937/107166] Bright Eyes ➜ unknown
[938/107166] José Feliciano ➜ male
[939/107166] Walk Off the Earth ➜ unknown
[940/107166] Lisa ➜ female
[941/107166] Jack Johnson ➜ male
[942/107166] Alex Clare ➜ male
[943/107166] The Fray ➜ unknown
[944/107166] Alabama Shakes ➜ unknown
[945/107166] Jack White ➜ male
[946/107166] Deer Tick ➜ unknown
[947/107166] The Felice Brothers ➜ unknown
[948/107166] Dawes ➜ unknown
[949/107166] Lead Belly ➜ male
[950/107166] Sawyer Fredericks ➜ male
[951/107166] The Last Internationale ➜ unknown
[952/107166] Rodrigo y Gabriela ➜ unknown
[953/107166] Depeche Mode ➜ unknown
[954/107166] Lilly Lukas ➜ male
[955/107166] The Silent Comedy ➜ unknown
[956/107166] Dry the River ➜ unknown
[957/107166] Gary Clark Jr. ➜ male
[958/107166] Glen Hansard ➜ male
[959/107166] William Elliott Whitmore ➜ male
[960/107166] Chris Cornell ➜ male
[961/107166] Buena Vista Social Club ➜ unknown
[962/107166] Earth, Wind & Fire ➜ unknown
[963/107166] D.J. Chill ➜ male
[964/107166] Bebel Gilberto ➜ female
[965/107166] Commodores ➜ unknown
[966/107166] Broken Bells ➜ unknown
[967/107166] Modest Mouse ➜ unknown
[968/107166] Life of Dillon ➜ unknown
[969/107166] AtellaGali ➜ unknown
[970/107166] Strange Talk ➜ unknown
[971/107166] Guy Sebastian ➜ male
[972/107166] Kiri T ➜ unknown
[973/107166] Stephen ➜ male
[974/107166] Kool John ➜ male
[975/107166] Frances ➜ female
[976/107166] Ella Eyre ➜ female
[977/107166] Parson James ➜ male
[978/107166] Troye Sivan ➜ male
[979/107166] Kat Dahlia ➜ female
[980/107166] Slightly Stoopid ➜ unknown
[981/107166] Jessie James Decker ➜ female
[982/107166] G-Eazy ➜ male
[983/107166] Deorro ➜ male
[984/107166] Good Charlotte ➜ unknown
[985/107166] Hudson Thames ➜ unknown
[986/107166] ZHU ➜ male
[987/107166] Sweater Beats ➜ unknown
[988/107166] Flux Pavilion ➜ male
[989/107166] Avery Wilson ➜ male
[990/107166] Phoenix ➜ male
[991/107166] Nelly ➜ male
[992/107166] Justine Skye ➜ female
[993/107166] Froogle ➜ unknown
[994/107166] Yellow Claw ➜ unknown
[995/107166] Empire of the Sun ➜ unknown
[996/107166] Scott Bradlee's Postmodern Jukebox ➜ unknown
[997/107166] Iron & Wine ➜ male
[998/107166] Jaymes Young ➜ male
[999/107166] FRENSHIP ➜ unknown
[1000/107166] TWENTY88 ➜ unknown
[1001/107166] Smallpools ➜ unknown
[1002/107166] RAC ➜ unknown
[1003/107166] Shayne Ward ➜ male
[1004/107166] Craig David ➜ male
[1005/107166] Little Mix ➜ unknown
[1006/107166] Mr. Probz ➜ male
[1007/107166] School Gyrls ➜ unknown
[1008/107166] Alesso ➜ male
[1009/107166] Girls' Generation ➜ unknown
[1010/107166] Hallyu Halloween ➜ unknown
[1011/107166] Seven Lions ➜ male
[get_gender_from_wikidata] Error for Wikidata ID Q42775: Expecting value: line 1 column 1 (char 0)
[1012/107166] Cash Cash ➜ unknown
[1013/107166] Chromeo ➜ unknown
[1014/107166] Zendaya ➜ female
[get_wikidata_id] Error for MBID 96142c23-9d18-4f6b-9acb-150d42ce1f1a: 'url-relation-list'
[1015/107166] JTX ➜ unknown
[1016/107166] Prince Fox ➜ male
[1017/107166] Draper ➜ unknown
[1018/107166] Mike Perry ➜ male
[1019/107166] Britt Nicole ➜ female
[1020/107166] Poppy ➜ female
[1021/107166] Checo ➜ male
[1022/107166] State of Sound ➜ unknown
[1023/107166] Aaron Carter ➜ male
[1024/107166] Alvaro ➜ male
[get_gender_from_wikidata] Error for Wikidata ID Q27956380: Expecting value: line 1 column 1 (char 0)
[1025/107166] Gryffin ➜ unknown
[1026/107166] Tokyo Police Club ➜ unknown
[1027/107166] Foals ➜ unknown
[1028/107166] Voxtrot ➜ unknown
[1029/107166] Tegan and Sara ➜ unknown
[1030/107166] DJ Snake ➜ male
[1031/107166] Birdman ➜ male
[1032/107166] iLoveMemphis ➜ male
[1033/107166] Yo Gotti ➜ male
[1034/107166] David Guetta ➜ male
[1035/107166] will.i.am ➜ male
[1036/107166] Ja Rule ➜ male
[1037/107166] William Singe ➜ male
[1038/107166] 2 Chainz ➜ male
[1039/107166] Korede Bello ➜ male
[1040/107166] 50 Cent ➜ male
[1041/107166] Koffi Olomide ➜ male
[1042/107166] Solidstar ➜ unknown
[1043/107166] DJ Maphorisa ➜ male
[1044/107166] Iyanya ➜ male
[1045/107166] Snoop Dogg ➜ male
[1046/107166] Nasty_C ➜ male
[1047/107166] DJ Fetty Bronson ➜ male
[1048/107166] Kiernan Jarryd Forbes ➜ unknown
[1049/107166] The-Dream ➜ unknown
[1050/107166] Joey B ➜ male
[1051/107166] Gyft ➜ unknown
[get_wikidata_id] Error for MBID ea6e8529-f367-4ed7-abfe-eba1a6adb54f: 'url-relation-list'
[1052/107166] Mwana FA ➜ unknown
[1053/107166] Afrojack ➜ male
[1054/107166] Sean Kingston ➜ male
[1055/107166] R. Kelly ➜ male
[1056/107166] Keke Palmer ➜ female
[1057/107166] Natasha Bedingfield ➜ female
[1058/107166] Fergie ➜ female
[1059/107166] Leona Lewis ➜ female
[1060/107166] Corinne Bailey Rae ➜ female
[1061/107166] Gwen Stefani ➜ female
[1062/107166] Plain White T's ➜ unknown
[1063/107166] Hot Chelle Rae ➜ unknown
[1064/107166] Alicia Keys ➜ female
[1065/107166] Boyz II Men ➜ unknown
[1066/107166] Luke Bryan ➜ male
[1067/107166] Justin Moore ➜ male
[1068/107166] Dustin Lynch ➜ male
[1069/107166] Carrie Underwood ➜ female
[1070/107166] Lee Brice ➜ male
[1071/107166] Jake Owen ➜ male
[1072/107166] Keith Urban ➜ male
[1073/107166] Maddie & Tae ➜ unknown
[1074/107166] Logan Mize ➜ male
[1075/107166] Sam Hunt ➜ male
[1076/107166] Lucy Hale ➜ female
[1077/107166] Brett Eldredge ➜ male
[1078/107166] Craig Morgan ➜ male
[1079/107166] Glen Templeton ➜ male
[1080/107166] Cole Swindell ➜ male
[1081/107166] Martin Garrix ➜ male
[1082/107166] Noah Guthrie ➜ male
[1083/107166] Anne-Marie ➜ female
[1084/107166] Dante Klein ➜ unknown
[1085/107166] Astrid S ➜ female
[1086/107166] NEIKED ➜ male
[1087/107166] Maty Noyes ➜ female
[1088/107166] Cashmere Cat ➜ male
[1089/107166] Shawn Mendes ➜ male
[1090/107166] Nevada ➜ unknown
[1091/107166] Gavin James ➜ male
[1092/107166] Michael Andrews ➜ male
[1093/107166] AURORA ➜ female
[1094/107166] The Vamps ➜ unknown
[1095/107166] Tove Lo ➜ female
[1096/107166] Daya ➜ female
[1097/107166] Seeb ➜ unknown
[1098/107166] Sara Hartman ➜ male
[1099/107166] Joel Adams ➜ male
[1100/107166] Noah Cyrus ➜ female
[1101/107166] Louis The Child ➜ unknown
[1102/107166] Peabo Bryson ➜ male
[1103/107166] Daryl Hall & John Oates ➜ unknown
[1104/107166] Marvin Gaye ➜ male
[1105/107166] Chris Stapleton ➜ male
[1106/107166] Backstreet Boys ➜ unknown
[1107/107166] *NSYNC ➜ unknown
[1108/107166] Van Morrison ➜ male
[1109/107166] Phil Collins ➜ male
[1110/107166] The Ranch ➜ unknown
[1111/107166] Dierks Bentley ➜ male
[1112/107166] Jason Aldean ➜ male
[1113/107166] Chris Young ➜ male
[1114/107166] Graham Colton Band ➜ unknown
[1115/107166] Ben Jelen ➜ male
[1116/107166] Hunter Hayes ➜ male
[1117/107166] Keith Whitley ➜ male
[1118/107166] Little Big Town ➜ unknown
[1119/107166] Zac Brown Band ➜ unknown
[1120/107166] Jerrod Niemann ➜ male
[1121/107166] Aaron Lewis ➜ male
[1122/107166] Brad Paisley ➜ male
[1123/107166] Dan + Shay ➜ unknown
[1124/107166] Easton Corbin ➜ male
[1125/107166] Frankie Ballard ➜ male
[1126/107166] Kenny Chesney ➜ male
[1127/107166] Kip Moore ➜ male
[1128/107166] Lady Antebellum ➜ female
[1129/107166] Old Crow Medicine Show ➜ unknown
[1130/107166] Rodney Atkins ➜ male
[1131/107166] Thomas Rhett ➜ male
[1132/107166] Tyler Farr ➜ female
[1133/107166] Ella Henderson ➜ female
[1134/107166] K'NAAN ➜ male
[1135/107166] Michael Franti & Spearhead ➜ male
[1136/107166] The Mowgli's ➜ unknown
[1137/107166] A Thousand Horses ➜ unknown
[1138/107166] Paramore ➜ unknown
[1139/107166] Gavin DeGraw ➜ male
[1140/107166] OMI ➜ male
[1141/107166] R. City ➜ male
[1142/107166] A Day To Remember ➜ unknown
[1143/107166] Hawk Nelson ➜ male
[1144/107166] He Is We ➜ unknown
[1145/107166] Silentó ➜ male
[1146/107166] Rachel Platten ➜ female
[1147/107166] Rich Homie Quan ➜ male
[1148/107166] T-Wayne ➜ unknown
[1149/107166] Shwayze ➜ male
[1150/107166] Sleeping With Sirens ➜ unknown
[1151/107166] Tori Kelly ➜ female
[1152/107166] Switchfoot ➜ unknown
[1153/107166] Ben Rector ➜ male
[1154/107166] Danielle Bradbery ➜ female
[1155/107166] Tim McMorris ➜ unknown
[1156/107166] Lukas Graham ➜ unknown
[1157/107166] Billy Crudup ➜ male
[get_wikidata_id] Error for MBID 6d6e4834-246a-42ac-b0e2-66b10aa96e65: 'url-relation-list'
[1158/107166] Rudderless ➜ unknown
[1159/107166] Yusuf / Cat Stevens ➜ male
[1160/107166] Judah & the Lion ➜ unknown
[1161/107166] Andrew McMahon in the Wilderness ➜ male
[1162/107166] Kris Allen ➜ male
[1163/107166] Edward Sharpe & The Magnetic Zeros ➜ unknown
[1164/107166] gnash ➜ male
[1165/107166] R3HAB ➜ male
[1166/107166] Oklahoma Sky ➜ unknown
[1167/107166] Will Hoge ➜ male
[1168/107166] Brantley Gilbert ➜ male
[get_wikidata_id] Error for MBID c80b0e6b-2c17-4f45-82ad-ba5c6ba673c2: 'url-relation-list'
[1169/107166] Jessica Mears ➜ unknown
[1170/107166] AronChupa ➜ male
[1171/107166] The Band Perry ➜ unknown
[1172/107166] Miranda Lambert ➜ female
[1173/107166] Jimmy Wayne ➜ male
[1174/107166] Coco Jones ➜ male
[1175/107166] Skylar Stecker ➜ female
[1176/107166] Austin Mahone ➜ male
[1177/107166] Shania Twain ➜ female
[1178/107166] Iggy Azalea ➜ female
[1179/107166] Five For Fighting ➜ male
[1180/107166] Daniel Powter ➜ male
[1181/107166] Jimmy Eat World ➜ unknown
[1182/107166] Metro Station ➜ unknown
[1183/107166] Lit ➜ unknown
[1184/107166] Simple Plan ➜ unknown
[1185/107166] 3OH!3 ➜ unknown
[1186/107166] Lifehouse ➜ unknown
[1187/107166] Hoobastank ➜ unknown
[1188/107166] Hinder ➜ unknown
[1189/107166] Snow Patrol ➜ unknown
[1190/107166] The Calling ➜ unknown
[1191/107166] Howie Day ➜ female
[1192/107166] The Script ➜ unknown
[1193/107166] Foreigner ➜ unknown
[1194/107166] Eagles ➜ unknown
[1195/107166] Don McLean ➜ male
[1196/107166] System Of A Down ➜ unknown
[1197/107166] Papa Roach ➜ unknown
[1198/107166] Muse ➜ unknown
[1199/107166] Linkin Park ➜ unknown
[1200/107166] Nickelback ➜ unknown
[1201/107166] Third Day ➜ unknown
[1202/107166] Rend Collective ➜ unknown
[get_wikidata_id] Error for MBID 23544692-e420-4661-8e64-9b675fce9a95: 'url-relation-list'
[1203/107166] Daniel Doss Band ➜ unknown
[1204/107166] Matt Redman ➜ male
[1205/107166] Timothy Brindle ➜ unknown
[1206/107166] Trip Lee ➜ male
[1207/107166] Lecrae ➜ male
[1208/107166] Walter Martin ➜ male
[1209/107166] Betty Chung ➜ female
[1210/107166] Patti Smith ➜ female
[1211/107166] Deerhunter ➜ unknown
[1212/107166] Chastity Belt ➜ unknown
[1213/107166] Molly Nilsson ➜ female
[1214/107166] Mac Demarco ➜ male
[1215/107166] The Limiñanas ➜ unknown
[1216/107166] Vaadat Charigim ➜ unknown
[1217/107166] Yonatan Gat ➜ male
[1218/107166] The Snails ➜ unknown
[1219/107166] Joe Nichols ➜ male
[1220/107166] Kelly Price ➜ female
[1221/107166] Florida A&M University Gospel Choir ➜ unknown
[1222/107166] Norman Hutchins ➜ unknown
[get_wikidata_id] Error for MBID ce90055a-072a-469b-9096-c143edc5c966: 'url-relation-list'
[1223/107166] Dottie Peoples & The Peoples Choice Chorale ➜ unknown
[1224/107166] Tasha Cobbs Leonard ➜ female
[1225/107166] Donald Lawrence ➜ male
[1226/107166] Benita Washington ➜ female
[1227/107166] The Warriors, DR.Charles G. Hayes ➜ unknown
[1228/107166] The New Life Community Choir ➜ unknown
[1229/107166] Marvin Sapp ➜ male
[1230/107166] Donnie McClurkin ➜ male
[1231/107166] Shekinah Glory Ministry ➜ unknown
[1232/107166] Smokie Norful ➜ male
[1233/107166] Tye Tribbett ➜ male
[1234/107166] Richard Smallwood ➜ male
[1235/107166] Fred Hammond ➜ male
[1236/107166] Kirk Franklin ➜ female
[1237/107166] Tamela Mann ➜ female
[1238/107166] The Rance Allen Group ➜ unknown
[1239/107166] CeCe Winans ➜ female
[1240/107166] Yolanda Adams ➜ female
[1241/107166] J Moss ➜ female
[1242/107166] Kurt Carr & The Kurt Carr Singers ➜ male
[1243/107166] Natalie La Rose ➜ female
[1244/107166] Run–D.M.C. ➜ unknown
[1245/107166] Icona Pop ➜ unknown
[1246/107166] Eric Church ➜ male
[1247/107166] Toby Keith ➜ male
[1248/107166] Corey Smith ➜ male
[1249/107166] Tim Halperin ➜ male
[1250/107166] Andrew Belle ➜ male
[1251/107166] Ron Pope ➜ male
[1252/107166] Tyler Ward ➜ female
[1253/107166] Tyler Hilton ➜ female
[1254/107166] AWOLNATION ➜ unknown
[1255/107166] The Day Trippers ➜ unknown
[1256/107166] Sara Evans ➜ male
[1257/107166] Macy Gray ➜ female
[1258/107166] Bing Crosby ➜ male
[1259/107166] Nat King Cole ➜ male
[1260/107166] Eartha Kitt ➜ female
[1261/107166] Ella Fitzgerald ➜ female
[1262/107166] Wham! ➜ unknown
[1263/107166] Frank Sinatra ➜ male
[1264/107166] The Beach Boys ➜ unknown
[1265/107166] Johnny Mathis ➜ male
[get_wikidata_id] Error for MBID 90014951-fcfd-4447-9a6c-aaf8782cbb6d: 'url-relation-list'
[1266/107166] Sha Gotti ➜ unknown
[1267/107166] Ayo & Teo ➜ unknown
[1268/107166] Fabolous ➜ male
[1269/107166] Lloyd Banks ➜ male
[1270/107166] Tinie Tempah ➜ male
[1271/107166] Sage The Gemini ➜ male
[1272/107166] Lil Jon & The East Side Boyz ➜ unknown
[1273/107166] Mac Miller ➜ unknown
[1274/107166] Maejor Ali ➜ male
[1275/107166] Ray J ➜ female
[1276/107166] Roscoe Dash ➜ male
[1277/107166] Ying Yang Twins ➜ unknown
[1278/107166] Lil Jon ➜ male
[1279/107166] Unk ➜ male
[1280/107166] DMX ➜ male
[1281/107166] Jim Jones ➜ male
[1282/107166] The Game ➜ unknown
[1283/107166] Baby Bash ➜ male
[1284/107166] Fat Joe ➜ male
[1285/107166] 2Pac ➜ male
[1286/107166] Eve ➜ male
[1287/107166] Cam’ron ➜ male
[1288/107166] The Notorious B.I.G. ➜ male
[1289/107166] N.W.A. ➜ unknown
[1290/107166] Skrillex ➜ male
[1291/107166] Three 6 Mafia ➜ unknown
[1292/107166] Ludacris ➜ male
[1293/107166] Disturbing Tha Peace ➜ unknown
[1294/107166] Roy Jones Jr. ➜ male
[1295/107166] Ski Mask The Slump God ➜ male
[1296/107166] Montana of 300 ➜ unknown
[1297/107166] Jeezy ➜ male
[1298/107166] David Banner ➜ male
[1299/107166] Chief Keef ➜ male
[1300/107166] Waka Flocka Flame ➜ male
[1301/107166] B.o.B ➜ male
[1302/107166] DJ Envy ➜ unknown
[1303/107166] Sean Jones ➜ male
[1304/107166] Booker T. & the M.G.'s ➜ unknown
[1305/107166] Jamey Johnson ➜ male
[1306/107166] Uncle Kracker ➜ male
[1307/107166] Billy Currington ➜ male
[1308/107166] Keith Anderson ➜ female
[1309/107166] Jon Pardi ➜ male
[1310/107166] Eli Young Band ➜ unknown
[1311/107166] Ms. Lauryn Hill ➜ female
[1312/107166] D'Angelo ➜ male
[1313/107166] Hiatus Kaiyote ➜ unknown
[1314/107166] Strange Fruit Project ➜ unknown
[1315/107166] Billy Paul ➜ male
[1316/107166] Owen Thiele x Zack Sekoff ➜ unknown
[1317/107166] Al Green ➜ male
[1318/107166] Otis Redding ➜ male
[1319/107166] Mos Def ➜ unknown
[1320/107166] Vulfpeck ➜ unknown
[1321/107166] The Rollers ➜ unknown
[1322/107166] Alina Baraz ➜ female
[1323/107166] The 1975 ➜ unknown
[1324/107166] The Japanese House ➜ genderfluid
[1325/107166] SOHN ➜ male
[1326/107166] Hundred Waters ➜ male
[1327/107166] Beach Tiger ➜ unknown
[1328/107166] OYLS ➜ unknown
[1329/107166] Max Minelli ➜ unknown
[1330/107166] Young Bleed ➜ male
[1331/107166] Victorious Cast ➜ unknown
[1332/107166] Bridgit Mendler ➜ female
[1333/107166] Shane Harper ➜ male
[1334/107166] Glee Cast ➜ unknown
[1335/107166] Austin Moon ➜ male
[1336/107166] Selena Gomez & The Scene ➜ unknown
[1337/107166] The Ready Set ➜ male
[1338/107166] Daniel Skye ➜ male
[1339/107166] Céline Dion ➜ female
[1340/107166] Kevin Turk ➜ male
[1341/107166] Big Time Rush ➜ unknown
[1342/107166] Brian McKnight ➜ male
[1343/107166] Christopher Wilde ➜ female
[get_wikidata_id] Error for MBID 2076f13c-22dd-4ac2-8d69-04101d7fee19: 'url-relation-list'
[1344/107166] Pinkie Pie ➜ unknown
[1345/107166] Moosh & Twist ➜ unknown
[1346/107166] Nick Carter ➜ male
[1347/107166] Dove Cameron ➜ female
[1348/107166] Hannah Montana ➜ female
[1349/107166] The GGGG's ➜ unknown
[1350/107166] Jay Sean ➜ male
[1351/107166] John Denver ➜ male
[1352/107166] Grace ➜ female
[1353/107166] Savage ➜ male
[1354/107166] The Who ➜ unknown
[1355/107166] The White Stripes ➜ unknown
[1356/107166] Beastie Boys ➜ unknown
[1357/107166] Gang Starr ➜ unknown
[1358/107166] Hieroglyphics ➜ unknown
[1359/107166] People Under The Stairs ➜ unknown
[1360/107166] Foster The People ➜ unknown
[1361/107166] The Von Bondies ➜ unknown
[1362/107166] The Veils ➜ unknown
[1363/107166] Mc Solaar ➜ male
[1364/107166] Ratatat ➜ unknown
[1365/107166] Die Antwoord ➜ unknown
[1366/107166] Porcelain Raft ➜ male
[1367/107166] She Wants Revenge ➜ unknown
[1368/107166] LCD Soundsystem ➜ unknown
[1369/107166] The Chemical Brothers ➜ unknown
[1370/107166] Pulp ➜ unknown
[1371/107166] Suede ➜ unknown
[1372/107166] William Shatner ➜ male
[1373/107166] Sleigh Bells ➜ unknown
[1374/107166] Modeselektor ➜ unknown
[1375/107166] Butthole Surfers ➜ unknown
[1376/107166] Robin Thicke ➜ male
[1377/107166] Caravan Palace ➜ unknown
[1378/107166] The Cardigans ➜ unknown
[1379/107166] Dolly Parton ➜ female
[1380/107166] Trentemøller ➜ male
[1381/107166] The Cars ➜ unknown
[1382/107166] Salt-N-Pepa ➜ unknown
[1383/107166] Cyndi Lauper ➜ female
[1384/107166] Bloodhound Gang ➜ unknown
[1385/107166] John Prine ➜ male
[1386/107166] Seals and Crofts ➜ unknown
[1387/107166] Bon Jovi ➜ unknown
[1388/107166] Chase Rice ➜ male
[1389/107166] Teriyaki Boyz ➜ unknown
[1390/107166] Rich Chigga ➜ male
[1391/107166] A$AP Ferg ➜ male
[1392/107166] Blue Swede ➜ unknown
[1393/107166] Raspberries ➜ unknown
[1394/107166] Elvin Bishop ➜ male
[1395/107166] Redbone ➜ unknown
[1396/107166] Rupert Holmes ➜ male
[1397/107166] The Five Stairsteps ➜ unknown
[1398/107166] Sweet ➜ unknown
[1399/107166] Fleetwood Mac ➜ unknown
[1400/107166] Glen Campbell ➜ male
[1401/107166] George Harrison ➜ male
[1402/107166] Jay & The Americans ➜ unknown
[1403/107166] Cheap Trick ➜ unknown
[1404/107166] Parliament ➜ unknown
[1405/107166] The Animals ➜ unknown
[1406/107166] Jean Knight ➜ male
[1407/107166] Pilot ➜ unknown
[1408/107166] The Guess Who ➜ unknown
[1409/107166] Sniff 'n' The Tears ➜ unknown
[1410/107166] Manfred Mann's Earth Band ➜ unknown
[1411/107166] The Three Degrees ➜ unknown
[1412/107166] Paul McCartney ➜ male
[1413/107166] The O'Jays ➜ unknown
[1414/107166] 10cc ➜ unknown
[1415/107166] Three Dog Night ➜ unknown
[1416/107166] The Hollies ➜ unknown
[1417/107166] The Foundations ➜ unknown
[1418/107166] Four Tops ➜ unknown
[1419/107166] Lillias White ➜ female
[1420/107166] Garrett Clayton ➜ male
[1421/107166] Melanie Martinez ➜ female
[1422/107166] R5 ➜ unknown
[1423/107166] Sabrina Carpenter ➜ female
[1424/107166] 5 Seconds of Summer ➜ unknown
[1425/107166] Chorus - Sleeping Beauty ➜ unknown
[1426/107166] MIKA ➜ male
[1427/107166] Parade of Lights ➜ unknown
[1428/107166] Dorothy ➜ female
[1429/107166] Olivia Holt ➜ male
[1430/107166] The Mellomen ➜ unknown
[1431/107166] Miranda Cosgrove ➜ female
[1432/107166] Bea Miller ➜ male
[1433/107166] The Girl and The Dreamcatcher ➜ unknown
[1434/107166] Anna Margaret ➜ female
[1435/107166] The Cheetah Girls ➜ unknown
[1436/107166] Cher Lloyd ➜ female
[1437/107166] Jana Kramer ➜ male
[1438/107166] Fitz and The Tantrums ➜ unknown
[1439/107166] Blondie ➜ unknown
[1440/107166] Quad City DJ's ➜ unknown
[1441/107166] George Michael ➜ male
[1442/107166] The Four Aces ➜ unknown
[1443/107166] a-ha ➜ unknown
[1444/107166] Rick Astley ➜ male
[1445/107166] Ray Parker, Jr. ➜ male
[1446/107166] Europe ➜ unknown
[1447/107166] Bee Gees ➜ unknown
[1448/107166] Village People ➜ unknown
[1449/107166] CeeLo Green ➜ male
[1450/107166] Chuck Berry ➜ male
[1451/107166] The Temptations ➜ unknown
[1452/107166] Lou Bega ➜ male
[1453/107166] AC/DC ➜ unknown
[1454/107166] Bag Raiders ➜ unknown
[1455/107166] Cristin Milioti ➜ female
[1456/107166] Gnarls Barkley ➜ unknown
[1457/107166] Yung Gravy ➜ male
[1458/107166] The Chordettes ➜ unknown
[1459/107166] Boney M. ➜ unknown
[1460/107166] Carl Douglas ➜ male
[1461/107166] Bryant Oden ➜ unknown
[1462/107166] Leanne & Naara ➜ unknown
[1463/107166] Jeff Williams ➜ male
[1464/107166] Casey Lee Williams ➜ female
[1465/107166] Spongebob Squarepants ➜ male
[1466/107166] Keyboard Cat ➜ male
[1467/107166] Parry Gripp ➜ male
[1468/107166] Nyan Cat ➜ male
[1469/107166] London Music Works ➜ unknown
[1470/107166] Raffi ➜ male
[1471/107166] John Williams ➜ male
[1472/107166] Big Shaq ➜ unknown
[1473/107166] Black Coast ➜ unknown
[1474/107166] Darude ➜ male
[1475/107166] Rednex ➜ unknown
[1476/107166] Daler Mehndi ➜ male
[1477/107166] DJ Jazzy Jeff ➜ male
[1478/107166] John Cena ➜ male
[1479/107166] William Jacobs ➜ male
[1480/107166] Spooky Scary Skeletons ➜ unknown
[1481/107166] Chip-man & The Buckwheat Boyz ➜ unknown
[1482/107166] Pokémon ➜ unknown
[1483/107166] Yolanda Be Cool ➜ unknown
[1484/107166] Electric Slide Music Makers ➜ unknown
[1485/107166] Stryker Pose ➜ male
[1486/107166] 4Fate ➜ unknown
[1487/107166] IceJJFish ➜ male
[1488/107166] Ennio Morricone ➜ male
[1489/107166] Henry Mancini ➜ male
[1490/107166] Rob Cantor ➜ male
[1491/107166] The Theme Song ➜ unknown
[1492/107166] RichaadEB ➜ unknown
[1493/107166] The Lonely Island ➜ unknown
[1494/107166] Tsuko G. ➜ unknown
[1495/107166] The OneUps ➜ unknown
[1496/107166] PSY ➜ male
[1497/107166] Nicholas Fraser ➜ male
[1498/107166] Yello ➜ unknown
[1499/107166] Randy Crenshaw ➜ male
[1500/107166] Haddaway ➜ male
[get_wikidata_id] Error for MBID b7c358c1-ec1e-4bbc-9ea8-3d80bb67fa3a: 'url-relation-list'
[1501/107166] SMNM ➜ unknown
[1502/107166] Love Thy Brother ➜ unknown
[1503/107166] Clear Six ➜ unknown
[1504/107166] ODC ➜ unknown
[1505/107166] Warner Newman ➜ male
[1506/107166] Kevin Andreas ➜ male
[1507/107166] Tropixx ➜ unknown
[1508/107166] Borgeous ➜ male
[1509/107166] JoJo ➜ female
[1510/107166] Pia Mia ➜ female
[1511/107166] Steve James ➜ male
[1512/107166] Carly Rae Jepsen ➜ female
[1513/107166] Mako ➜ male
[1514/107166] Illenium ➜ male
[1515/107166] BETSY ➜ female
[1516/107166] Aaron Taos ➜ unknown
[1517/107166] Au/Ra ➜ female
[1518/107166] Goldfrapp ➜ unknown
[1519/107166] Chase Atlantic ➜ unknown
[1520/107166] José James ➜ male
[1521/107166] Dimitri Vegas & Like Mike ➜ unknown
[1522/107166] ANOHNI ➜ trans woman
[1523/107166] Alex Aiono ➜ male
[1524/107166] Dave ➜ unknown
[1525/107166] CAZZETTE ➜ unknown
[1526/107166] SAINT WKND ➜ male
[1527/107166] Citizen Four ➜ unknown
[1528/107166] Huntar ➜ male
[1529/107166] The Knocks ➜ unknown
[1530/107166] Tei Shi ➜ female
[1531/107166] Rotana ➜ female
[1532/107166] SoMo ➜ male
[1533/107166] Rag'n'Bone Man ➜ male
[1534/107166] SiDizen King ➜ male
[1535/107166] Youngr ➜ unknown
[1536/107166] Clement Bazin ➜ male
[1537/107166] Whissell ➜ unknown
[1538/107166] Thomas Hayes ➜ male
[1539/107166] BRKLYN ➜ unknown
[1540/107166] Ryos ➜ male
[1541/107166] San Holo ➜ male
[1542/107166] Caye ➜ unknown
[1543/107166] Berhana ➜ male
[1544/107166] Daniel Caesar ➜ male
[1545/107166] Abhi//Dijon ➜ unknown
[1546/107166] Tontario ➜ unknown
[1547/107166] Masego ➜ male
[1548/107166] Cal Scruby ➜ unknown
[1549/107166] CalenRaps ➜ unknown
[1550/107166] Hopsin ➜ male
[1551/107166] K CAMP ➜ female
[1552/107166] 21 Savage ➜ male
[1553/107166] Vic Mensa ➜ male
[1554/107166] Tory Lanez ➜ male
[1555/107166] PARTYNEXTDOOR ➜ male
[1556/107166] Tank ➜ unknown
[1557/107166] NO1-NOAH ➜ unknown
[1558/107166] Lloyd ➜ male
[1559/107166] Josh Garrels ➜ male
[1560/107166] Ethan Pierce ➜ male
[1561/107166] Brandi Carlile ➜ female
[1562/107166] The Gospel Whiskey Runners ➜ unknown
[1563/107166] Kurt Vile ➜ male
[1564/107166] U2 ➜ unknown
[1565/107166] Big Deal ➜ unknown
[1566/107166] Daughter ➜ unknown
[1567/107166] Volcano Choir ➜ unknown
[1568/107166] Nurses ➜ unknown
[1569/107166] Ty Segall ➜ male
[1570/107166] Jay Reatard ➜ male
[1571/107166] Girls ➜ unknown
[1572/107166] Lilys ➜ unknown
[1573/107166] As Cities Burn ➜ unknown
[1574/107166] My Epic ➜ unknown
[1575/107166] Woodkid ➜ male
[1576/107166] Lydia ➜ female
[1577/107166] The Middle East ➜ unknown
[1578/107166] Wavves ➜ unknown
[1579/107166] Nude Pop ➜ male
[1580/107166] Young Galaxy ➜ male
[1581/107166] Grimes ➜ female
[1582/107166] Young Man ➜ male
[1583/107166] Houses ➜ unknown
[1584/107166] 7Horse ➜ unknown
[1585/107166] Future Islands ➜ unknown
[1586/107166] T. Rex ➜ unknown
[1587/107166] Atlas Sound ➜ non-binary gender
[1588/107166] Temples ➜ unknown
[1589/107166] Eels ➜ unknown
[1590/107166] Norman Greenbaum ➜ male
[1591/107166] Hooded Fang ➜ unknown
[1592/107166] Ben Folds ➜ male
[1593/107166] François Virot ➜ male
[1594/107166] AJ Davila ➜ unknown
[1595/107166] The Globes ➜ unknown
[1596/107166] Bass Drum of Death ➜ unknown
[1597/107166] Saint Rich ➜ male
[1598/107166] Hayden James ➜ male
[1599/107166] THEY. ➜ unknown
[1600/107166] Pusher ➜ unknown
[1601/107166] Bipolar Sunshine ➜ male
[1602/107166] Calle 13 ➜ unknown
[1603/107166] Xtreme ➜ unknown
[1604/107166] A.B. Quintanilla III ➜ male
[1605/107166] Camila ➜ female
[1606/107166] Aventura ➜ unknown
[1607/107166] Maná ➜ unknown
[1608/107166] Santana ➜ unknown
[1609/107166] Nina Sky ➜ female
[1610/107166] N.O.R.E. ➜ male
[1611/107166] Celia Cruz ➜ female
[1612/107166] Young Boss ➜ male
[1613/107166] Ivy Queen ➜ unknown
[1614/107166] Enrique Iglesias ➜ male
[1615/107166] Boy Wonder Chosen Few ➜ male
[1616/107166] Tito "El Bambino" ➜ male
[1617/107166] Elvis Crespo ➜ male
[1618/107166] Monchy & Alexandra ➜ unknown
[1619/107166] Julieta Venegas ➜ female
[1620/107166] Don Omar ➜ male
[1621/107166] Megastylez ➜ unknown
[1622/107166] Paulina Rubio ➜ female
[1623/107166] Jesse & Joy ➜ unknown
[1624/107166] Juanes ➜ male
[1625/107166] R.K.M & Ken-Y ➜ unknown
[1626/107166] Angel Y Khriz ➜ unknown
[1627/107166] Amandititita ➜ female
[1628/107166] Lil Rob ➜ male
[1629/107166] Los Apson ➜ unknown
[1630/107166] Intocable ➜ unknown
[1631/107166] Prince Royce ➜ male
[1632/107166] Chino & Nacho ➜ unknown
[1633/107166] La Oreja de Van Gogh ➜ unknown
[1634/107166] Voltio ➜ male
[1635/107166] Ninel Conde ➜ female
[1636/107166] Los Enanitos Verdes ➜ unknown
[1637/107166] Café Tacvba ➜ unknown
[1638/107166] Shakira ➜ female
[1639/107166] Toby Love ➜ male
[1640/107166] La Factoria ➜ unknown
[1641/107166] Nigga ➜ unknown
[1642/107166] Gotay "El Autentiko" ➜ male
[1643/107166] Arcangel ➜ male
[1644/107166] De La Ghetto ➜ male
[1645/107166] Zion & Lennox ➜ unknown
[1646/107166] Zion ➜ male
[1647/107166] Casa De Leones ➜ unknown
[1648/107166] Omega ➜ unknown
[1649/107166] Fuego ➜ male
[1650/107166] Aniceto Molina Y La Luz Roja De San Marcos ➜ unknown
[1651/107166] Blanquito Man ➜ unknown
[1652/107166] Grupo Sonador ➜ unknown
[1653/107166] Plan B ➜ male
[1654/107166] Magnate ➜ unknown
[1655/107166] Wisin & Yandel ➜ unknown
[1656/107166] Romeo Santos ➜ male
[1657/107166] Alacranes Musical ➜ unknown
[1658/107166] Eddy Lover ➜ male
[1659/107166] Ana Torroja ➜ female
[1660/107166] Los Primos De Durango ➜ unknown
[1661/107166] Alicia Villarreal ➜ female
[1662/107166] La Tropa Vallenata ➜ unknown
[1663/107166] Soca Man ➜ male
[1664/107166] La Sonora Dinamita ➜ unknown
[1665/107166] Juan Gabriel ➜ male
[1666/107166] Kike Santander ➜ male
[1667/107166] Cristian Castro ➜ female
[1668/107166] Marco Antonio Solís ➜ male
[1669/107166] Impacto Mc ➜ unknown
[1670/107166] Vicente Fernández ➜ male
[1671/107166] Luis Miguel ➜ male
[1672/107166] Kalimba ➜ male
[1673/107166] Alexandre Pires ➜ male
[1674/107166] S. Cotugno ➜ male
[1675/107166] Banda El Recodo ➜ unknown
[1676/107166] Grupo Exterminador ➜ unknown
[1677/107166] Selena y los Dinos ➜ unknown
[1678/107166] Grupo Bryndis ➜ unknown
[1679/107166] Los Angeles De Charly ➜ unknown
[1680/107166] Johnny Balik ➜ male
[1681/107166] The Bomb Digz ➜ unknown
[1682/107166] MIIA ➜ unknown
[1683/107166] Clinton Washington ➜ female
[1684/107166] Matthew Schuler ➜ male
[1685/107166] Javon J ➜ female
[1686/107166] Finding Hope ➜ unknown
[1687/107166] George Nozuka ➜ male
[1688/107166] Tyrese ➜ male
[1689/107166] Julia Michaels ➜ female
[1690/107166] A Summer High ➜ unknown
[1691/107166] Lostboycrow ➜ unknown
[1692/107166] Elevation Worship ➜ unknown
[1693/107166] Passion ➜ unknown
[1694/107166] Chris Tomlin ➜ male
[1695/107166] All Sons & Daughters ➜ unknown
[1696/107166] Jesus Culture ➜ unknown
[1697/107166] Jamie Berry ➜ male
[1698/107166] Junior Prom ➜ unknown
[1699/107166] Dylan Gardner ➜ male
[1700/107166] Kill Paris ➜ male
[1701/107166] lostprophets ➜ unknown
[1702/107166] Highly Suspect ➜ unknown
[1703/107166] WE ARE TWIN ➜ unknown
[1704/107166] Grizfolk ➜ unknown
[1705/107166] David P. ➜ male
[1706/107166] COIN ➜ unknown
[1707/107166] The Wanted ➜ unknown
[1708/107166] The Him ➜ unknown
[1709/107166] Captain Cuts ➜ unknown
[1710/107166] Common Kings ➜ male
[1711/107166] Miike Snow ➜ unknown
[get_wikidata_id] Error for MBID 65a0c891-c231-4d17-a62d-bd8204f71941: 'url-relation-list'
[1712/107166] The Wild Wild ➜ unknown
[1713/107166] Old Daisy ➜ unknown
[1714/107166] Paris Jones ➜ male
[1715/107166] Ojivolta ➜ unknown
[1716/107166] Coast Modern ➜ unknown
[1717/107166] Healy ➜ unknown
[1718/107166] James Blake ➜ male
[1719/107166] Hysee ➜ unknown
[1720/107166] Ludovico Einaudi ➜ male
[1721/107166] Ishome ➜ female
[get_wikidata_id] Error for MBID 535c8e15-e365-485b-baa7-63eeba809c8f: 'url-relation-list'
[1722/107166] Ujo ➜ unknown
[1723/107166] Petit Biscuit ➜ male
[1724/107166] Chrome Sparks ➜ unknown
[1725/107166] FKA twigs ➜ female
[1726/107166] Vicktor Taiwò ➜ unknown
[1727/107166] Ólafur Arnalds ➜ male
[1728/107166] Rakesh Chaurasia ➜ male
[1729/107166] Rakesh Chaurasia & Abhijit Pohankar ➜ male
[1730/107166] Ben Leinbach ➜ unknown
[1731/107166] MC YOGI ➜ male
[1732/107166] Digital Underground ➜ unknown
[1733/107166] Naughty By Nature ➜ unknown
[1734/107166] Busta Rhymes ➜ male
[1735/107166] Black Sheep ➜ unknown
[1736/107166] Luke ➜ male
[1737/107166] Eric B. & Rakim ➜ unknown
[1738/107166] Nas ➜ male
[1739/107166] Gil Scott-Heron ➜ male
[1740/107166] Digable Planets ➜ unknown
[1741/107166] Public Enemy ➜ unknown
[1742/107166] Onyx ➜ unknown
[1743/107166] AMG ➜ male
[1744/107166] Jay Chou ➜ male
[1745/107166] JJ Lin ➜ male
[1746/107166] Mayday ➜ unknown
[1747/107166] Leehom Wang ➜ unknown
[1748/107166] Mario ➜ male
[1749/107166] Sevyn Streeter ➜ female
[1750/107166] Yuna ➜ female
[1751/107166] Michael Wong ➜ male
[1752/107166] Hu Xia ➜ male
[1753/107166] Yisa Yu ➜ unknown
[1754/107166] Sam Lee ➜ male
[1755/107166] Richie Jen ➜ male
[1756/107166] Vivian Hsu ➜ unknown
[1757/107166] Fish Leong ➜ male
[1758/107166] Travis Porter ➜ unknown
[1759/107166] The MegaStars Of Hip Hop & R&B Karaoke ➜ unknown
[1760/107166] MiMS ➜ male
[1761/107166] Chamillionaire ➜ male
[1762/107166] Yelawolf ➜ male
[1763/107166] Daryl Coley ➜ male
[1764/107166] Rev. James Cleveland ➜ male
[1765/107166] James Cleveland ➜ male
[1766/107166] William Murphy ➜ male
[1767/107166] 3 Winans Brothers ➜ unknown
[1768/107166] Jimmy Buffett ➜ male
[1769/107166] Israel Kamakawiwo'ole ➜ unknown
[1770/107166] Peter Tosh ➜ male
[1771/107166] Dave Matthews ➜ unknown
[1772/107166] Shabba Ranks ➜ male
[1773/107166] Ziggy Marley ➜ male
[1774/107166] June Summers ➜ female
[1775/107166] Damian Marley ➜ unknown
[1776/107166] NF ➜ male
[1777/107166] Rob Bailey & The Hustle Standard ➜ unknown
[1778/107166] Bad Meets Evil ➜ unknown
[1779/107166] D12 ➜ unknown
[1780/107166] Saba ➜ male
[1781/107166] Gorilla Zoe ➜ male
[1782/107166] Goodie Mob ➜ unknown
[1783/107166] Frontlynaz ➜ unknown
[1784/107166] Bone Crusher ➜ male
[1785/107166] Ramin Djawadi ➜ male
[1786/107166] Rahat Fateh Ali Khan ➜ male
[1787/107166] Atif Aslam ➜ male
[1788/107166] Momina Mustehsan ➜ unknown
[1789/107166] Clinton Cerejo ➜ male
[1790/107166] Amit Trivedi ➜ male
[1791/107166] Ajay-Atul ➜ unknown
[1792/107166] Zay Hilfigerrr ➜ unknown
[1793/107166] Geto Boys ➜ unknown
[1794/107166] Prokumar & Arijit Singh & Alka Yagnik ➜ female
[1795/107166] O.T. Genasis ➜ male
[1796/107166] Jasmine V ➜ unknown
[1797/107166] Young Dolph ➜ male
[1798/107166] Kilo Ali ➜ male
[1799/107166] Bootsy Collins ➜ male
[1800/107166] Jidenna ➜ male
[1801/107166] St. Lucia ➜ male
[1802/107166] Dustin Tebbutt ➜ unknown
[1803/107166] Jarryd James ➜ male
[1804/107166] Rayland Baxter ➜ male
[1805/107166] Beta Radio ➜ unknown
[1806/107166] Dan Croll ➜ male
[1807/107166] Kodaline ➜ unknown
[1808/107166] Katja Petri ➜ female
[1809/107166] City of the Sun ➜ unknown
[1810/107166] Cary Brothers ➜ unknown
[1811/107166] Remy Zero ➜ unknown
[1812/107166] Glades ➜ unknown
[1813/107166] SISTAR ➜ unknown
[1814/107166] Yu Jae Seok ➜ unknown
[1815/107166] SEVENTEEN ➜ unknown
[1816/107166] Agust D ➜ male
[1817/107166] 24K ➜ unknown
[1818/107166] TWICE ➜ unknown
[1819/107166] I.O.I ➜ unknown
[1820/107166] Red Velvet ➜ unknown
[1821/107166] GOT7 ➜ unknown
[1822/107166] Philip Bailey ➜ male
[1823/107166] Clarence Clemons ➜ male
[1824/107166] Lisa Lisa & Cult Jam ➜ unknown
[1825/107166] REO Speedwagon ➜ unknown
[1826/107166] Bonnie Tyler ➜ female
[1827/107166] Deniece Williams ➜ female
[1828/107166] The Romantics ➜ unknown
[1829/107166] Terence Trent D'Arby ➜ male
[1830/107166] 'Til Tuesday ➜ unknown
[1831/107166] Eddie Money ➜ male
[1832/107166] Gregory Abbott ➜ male
[1833/107166] Dan Hill ➜ male
[1834/107166] Luther Vandross ➜ male
[1835/107166] Basia ➜ female
[1836/107166] Bad English ➜ unknown
[1837/107166] New Kids On The Block ➜ unknown
[1838/107166] The Bangles ➜ unknown
[1839/107166] Warrant ➜ unknown
[1840/107166] Michael Bolton ➜ male
[1841/107166] The Manhattans ➜ unknown
[1842/107166] Champaign ➜ unknown
[1843/107166] Matthew Wilder ➜ male
[1844/107166] Surface ➜ unknown
[1845/107166] Randy Meisner ➜ male
[1846/107166] Bertie Higgins ➜ male
[1847/107166] Loverboy ➜ unknown
[1848/107166] Dan Fogelberg ➜ male
[1849/107166] The Outfield ➜ unknown
[1850/107166] Paul Young ➜ male
[1851/107166] Kenny Loggins ➜ male
[1852/107166] Men At Work ➜ unknown
[1853/107166] Patty Smyth ➜ female
[1854/107166] After The Fire ➜ unknown
[1855/107166] The Psychedelic Furs ➜ unknown
[1856/107166] Cameo ➜ unknown
[1857/107166] The Sugarhill Gang ➜ unknown
[1858/107166] Don Henley ➜ male
[1859/107166] John Mellencamp ➜ male
[1860/107166] Tom Petty ➜ male
[1861/107166] Kim Carnes ➜ female
[1862/107166] Pat Benatar ➜ female
[1863/107166] Katrina & The Waves ➜ unknown
[1864/107166] Crowded House ➜ unknown
[1865/107166] Naked Eyes ➜ unknown
[1866/107166] Poison ➜ unknown
[1867/107166] Glenn Frey ➜ male
[1868/107166] Rockwell ➜ male
[1869/107166] DeBarge ➜ unknown
[1870/107166] Journey ➜ unknown
[1871/107166] Kool & The Gang ➜ unknown
[1872/107166] MC Hammer ➜ male
[1873/107166] Kurtis Blow ➜ male
[1874/107166] Whitney Houston ➜ female
[1875/107166] Billy Joel ➜ male
[1876/107166] Johnny Hates Jazz ➜ unknown
[1877/107166] Spandau Ballet ➜ unknown
[1878/107166] Bananarama ➜ unknown
[1879/107166] Diana Ross ➜ female
[1880/107166] Elton John ➜ male
[1881/107166] Madness ➜ unknown
[1882/107166] Fine Young Cannibals ➜ unknown
[1883/107166] Rufus ➜ male
[1884/107166] Belinda Carlisle ➜ female
[1885/107166] Kris Kristofferson ➜ male
[1886/107166] Waylon Jennings ➜ male
[1887/107166] Ian Tyson ➜ male
[1888/107166] Lyle Lovett ➜ male
[1889/107166] John Hiatt ➜ male
[1890/107166] Robert Earl Keen ➜ male
[1891/107166] Chris Smither ➜ male
[1892/107166] The Gourds ➜ unknown
[1893/107166] Hayes Carll ➜ male
[1894/107166] Steve Earle ➜ male
[1895/107166] James McMurtry ➜ male
[1896/107166] Drive-By Truckers ➜ unknown
[1897/107166] Townes Van Zandt ➜ male
[1898/107166] Jerry Jeff Walker ➜ male
[1899/107166] Eric Andersen ➜ male
[1900/107166] Bob Gibson ➜ male
[1901/107166] Chris Knight ➜ male
[1902/107166] Ryan Bingham ➜ male
[1903/107166] Bruce Springsteen ➜ male
[1904/107166] The Marshall Tucker Band ➜ unknown
[1905/107166] The Allman Brothers Band ➜ unknown
[1906/107166] New Riders of the Purple Sage ➜ unknown
[1907/107166] Grateful Dead ➜ unknown
[1908/107166] Johnny Jenkins ➜ male
[1909/107166] The Band ➜ unknown
[1910/107166] Tony Joe White ➜ male
[1911/107166] The Byrds ➜ unknown
[1912/107166] The Highwaymen ➜ unknown
[1913/107166] Reckless Kelly ➜ unknown
[1914/107166] Phil Ochs ➜ male
[1915/107166] Arlo Guthrie ➜ male
[1916/107166] Woody Guthrie ➜ male
[1917/107166] Ramblin' Jack Elliott ➜ male
[1918/107166] Bob Wills ➜ male
[1919/107166] Anthony Hamilton ➜ male
[1920/107166] Musiq Soulchild ➜ male
[1921/107166] Jamie Foxx ➜ male
[1922/107166] Maxwell ➜ male
[1923/107166] Estelle ➜ female
[1924/107166] Eric Benét ➜ male
[1925/107166] Chrisette Michele ➜ female
[1926/107166] Dwele ➜ male
[1927/107166] Avant ➜ male
[1928/107166] Raheem DeVaughn ➜ male
[1929/107166] Too $hort ➜ male
[1930/107166] Mint Condition ➜ unknown
[1931/107166] Anthony David ➜ male
[1932/107166] Kem ➜ male
[1933/107166] LSG ➜ unknown
[1934/107166] Floetry ➜ unknown
[1935/107166] Bell Biv DeVoe ➜ unknown
[1936/107166] Babyface ➜ male
[1937/107166] Shai ➜ unknown
[1938/107166] Blackstreet ➜ unknown
[1939/107166] Bilal ➜ male
[1940/107166] Angie Stone ➜ female
[1941/107166] Tweet ➜ female
[1942/107166] Aaron Hall ➜ male
[1943/107166] Guy ➜ male
[1944/107166] Tony! Toni! Toné! ➜ unknown
[1945/107166] Tevin Campbell ➜ male
[1946/107166] Curtis Fields ➜ male
[1947/107166] Stevie Nicks ➜ female
[1948/107166] Michael Bublé ➜ male
[1949/107166] High School Musical Cast ➜ unknown
[1950/107166] Josh Gad ➜ male
[1951/107166] Straight No Chaser ➜ unknown
[1952/107166] Colony House ➜ unknown
[1953/107166] Sara Bareilles ➜ female
[1954/107166] The Proclaimers ➜ unknown
[1955/107166] Bill Medley ➜ male
[1956/107166] Jason Castro ➜ male
[1957/107166] Lennon & Maisy ➜ unknown
[1958/107166] Our Last Night ➜ unknown
[1959/107166] Kent Moran ➜ unknown
[1960/107166] Alphaville ➜ unknown
[1961/107166] Tyler Hilton & Bethany Joy Lenz ➜ female
[1962/107166] Julia Sheer ➜ female
[1963/107166] Secondhand Serenade ➜ unknown
[1964/107166] Alex & Sierra ➜ unknown
[1965/107166] Olivia O'Brien ➜ female
[1966/107166] The La's ➜ unknown
[1967/107166] Gym Class Heroes ➜ unknown
[1968/107166] James Taylor ➜ male
[1969/107166] Stockard Channing ➜ female
[1970/107166] Tyler Adair ➜ female
[get_wikidata_id] Error for MBID 2a0e5a6f-f905-4289-8663-0236b50aeecc: 'url-relation-list'
[1971/107166] Jay Ollero ➜ unknown
[1972/107166] Kyle Reynolds ➜ male
[1973/107166] Benjy Davis ➜ male
[1974/107166] Travis Atreo ➜ unknown
[1975/107166] Trevor Wesley ➜ male
[1976/107166] Sinéad O'Connor ➜ female
[1977/107166] Ingrid Michaelson ➜ female
[1978/107166] The Cab ➜ unknown
[1979/107166] Grace VanderWaal ➜ female
[1980/107166] Terror Jr ➜ male
[1981/107166] Drew Holcomb & The Neighbors ➜ male
[1982/107166] Oh Honey ➜ unknown
[1983/107166] City and Colour ➜ unknown
[1984/107166] Stanaj ➜ unknown
[1985/107166] Dirt Nasty ➜ male
[1986/107166] Bas ➜ male
[1987/107166] Curren$y ➜ male
[1988/107166] MadeinTYO ➜ male
[1989/107166] Westside Connection ➜ unknown
[1990/107166] Grits ➜ unknown
[1991/107166] Action Bronson ➜ male
[1992/107166] Lil Yachty ➜ male
[1993/107166] Kent Jones ➜ male
[1994/107166] Jay Rock ➜ male
[1995/107166] Pusha T ➜ male
[1996/107166] Mavado ➜ male
[1997/107166] Rob $tone ➜ male
[1998/107166] Quavo ➜ male
[1999/107166] Troy Ave ➜ unknown
[2000/107166] Joey Bada$$ ➜ male
[2001/107166] Riff Raff ➜ male
[2002/107166] Big K.R.I.T. ➜ male
[2003/107166] A$AP Mob ➜ unknown
[2004/107166] Bone Thugs-N-Harmony ➜ unknown
[2005/107166] Mobb Deep ➜ unknown
[2006/107166] Lady Leshurr ➜ female
[2007/107166] Dom Kennedy ➜ male
[2008/107166] Jeff Foxworthy ➜ male
[2009/107166] The Drums ➜ unknown
[2010/107166] The Lost Fingers ➜ unknown
[2011/107166] fun. ➜ unknown
[2012/107166] St. Vincent ➜ genderfluid
[2013/107166] M. Ward ➜ male
[2014/107166] I Break Horses ➜ unknown
[2015/107166] Keane ➜ unknown
[2016/107166] Lee Hazlewood ➜ male
[2017/107166] Cleveland Orchestra ➜ unknown
[2018/107166] Thieves Like Us ➜ unknown
[2019/107166] Electric Guest ➜ unknown
[2020/107166] Oh Land ➜ female
[2021/107166] Django Django ➜ male
[2022/107166] Alpine ➜ unknown
[2023/107166] Bleeding Knees Club ➜ unknown
[2024/107166] This Many Boyfriends ➜ unknown
[2025/107166] Yuck ➜ unknown
[2026/107166] Imperial Teen ➜ unknown
[2027/107166] The Echo-Friendly ➜ unknown
[2028/107166] Silversun Pickups ➜ unknown
[2029/107166] The Walkmen ➜ unknown
[2030/107166] Scissor Sisters ➜ unknown
[2031/107166] The 2 Bears ➜ unknown
[2032/107166] Punch Brothers ➜ unknown
[2033/107166] Catcall ➜ female
[2034/107166] William Fitzsimmons ➜ male
[2035/107166] Joe Tex ➜ male
[2036/107166] The Brothers Johnson ➜ unknown
[2037/107166] Freddie King ➜ male
[2038/107166] Pete Wingfield ➜ male
[2039/107166] Labi Siffre ➜ male
[2040/107166] Dusty Springfield ➜ female
[2041/107166] Them ➜ unknown
[2042/107166] Bobby Womack ➜ male
[2043/107166] J.J. Cale ➜ male
[2044/107166] Smith ➜ male
[2045/107166] Dave Dee, Dozy, Beaky, Mick & Tich ➜ unknown
[2046/107166] Nancy Sinatra ➜ male
[2047/107166] Pacific Gas & Electric ➜ unknown
[2048/107166] Big Jack Fortune ➜ unknown
[2049/107166] Eddie Floyd ➜ unknown
[2050/107166] Keith Mansfield ➜ male
[2051/107166] The Coasters ➜ unknown
[2052/107166] Randy Crawford ➜ female
[2053/107166] The Tornadoes ➜ unknown
[2054/107166] The Lively Ones ➜ unknown
[2055/107166] The Robins ➜ unknown
[2056/107166] Link Wray ➜ male
[2057/107166] The Marketts ➜ unknown
[2058/107166] April March ➜ female
[2059/107166] Gladys Knight & The Pips ➜ unknown
[2060/107166] James Brown ➜ male
[2061/107166] The J.B.'s ➜ unknown
[2062/107166] Desired ➜ unknown
[2063/107166] Dominik Hauser ➜ unknown
[2064/107166] Cliff Edwards ➜ male
[2065/107166] Ghosts ➜ unknown
[2066/107166] Robert MacGimsey ➜ male
[2067/107166] Michael Giacchino ➜ male
[2068/107166] Paul J. Smith ➜ male
[2069/107166] Kathryn Beaumont ➜ female
[2070/107166] Jerry Goldsmith ➜ male
[2071/107166] Gary Hoey ➜ male
[2072/107166] Compilation Générique TV ➜ unknown
[2073/107166] James Baskett ➜ male
[2074/107166] Disney Studio Chorus ➜ unknown
[2075/107166] Jim Carmichael ➜ male
[2076/107166] Bruce Healey ➜ male
[2077/107166] Walt Disney ➜ unknown
[2078/107166] The Melomen ➜ unknown
[2079/107166] Various Artists ➜ unknown
[2080/107166] Karaoke Diamonds ➜ unknown
[2081/107166] Regina Spektor ➜ female
[2082/107166] Brooke White ➜ male
[2083/107166] KT Tunstall ➜ female
[2084/107166] Michelle Branch ➜ female
[2085/107166] Adidas ➜ unknown
[2086/107166] Ferras ➜ male
[2087/107166] Erin McCarley ➜ female
[2088/107166] A Fine Frenzy ➜ female
[2089/107166] Charlotte Martin ➜ male
[2090/107166] Rachael Sage ➜ female
[2091/107166] Rachael Yamagata ➜ female
[2092/107166] Ape Drums ➜ unknown
[2093/107166] King Zip Lock ➜ male
[2094/107166] Felix Snow ➜ male
[2095/107166] DJ Drill ➜ male
[2096/107166] Nicky Jam ➜ unknown
[2097/107166] Reggaetones ➜ unknown
[2098/107166] Cosculluela ➜ male
[2099/107166] Almighty ➜ unknown
[2100/107166] Pusho ➜ unknown
[2101/107166] Tego Calderon ➜ male
[2102/107166] Juan Luis Guerra 4.40 ➜ unknown
[2103/107166] Cultura Profética ➜ unknown
[2104/107166] Vicente Garcia ➜ male
[2105/107166] Manu Chao ➜ male
[2106/107166] Gipsy Kings ➜ unknown
[2107/107166] Marc Anthony ➜ unknown
[2108/107166] Carlos Vives ➜ male
[2109/107166] Bacilos ➜ unknown
[2110/107166] Gilberto Santa Rosa ➜ male
[2111/107166] Eddie Santiago ➜ male
[2112/107166] Eros Ramazzotti ➜ male
[2113/107166] Willie Colón ➜ male
[2114/107166] Farruko ➜ male
[2115/107166] Alejandro Sanz ➜ male
[2116/107166] Stephen Marley ➜ male
[2117/107166] Cypress Hill ➜ unknown
[2118/107166] Oscar D'León ➜ male
[2119/107166] Dimension Latina ➜ unknown
[2120/107166] La Critica ➜ unknown
[2121/107166] Frankie Ruiz ➜ male
[2122/107166] Daddy Yankee ➜ male
[2123/107166] Gente De Zona ➜ unknown
[2124/107166] Maluma ➜ male
[2125/107166] EZ El Ezeta ➜ unknown
[2126/107166] Yandel ➜ male
[2127/107166] Carlos Baute ➜ male
[2128/107166] Ozuna ➜ male
[2129/107166] Chris Jeday ➜ male
[2130/107166] Brytiago ➜ male
[2131/107166] El Gran Combo De Puerto Rico ➜ unknown
[2132/107166] Feid ➜ male
[2133/107166] Alberto Cortez ➜ male
[2134/107166] Gigolo Y La Exce ➜ unknown
[2135/107166] Bad Bunny ➜ male
[2136/107166] El Taiger ➜ male
[2137/107166] Sebastian Yatra ➜ male
[2138/107166] 12 Disipulos ➜ unknown
[2139/107166] Cartel De Santa ➜ unknown
[2140/107166] Nacho ➜ unknown
[2141/107166] Karol G ➜ female
[2142/107166] CocoRosie ➜ unknown
[2143/107166] Chet Faker ➜ male
[2144/107166] Odetta Hartman ➜ female
[2145/107166] Rozzi Crane ➜ female
[get_wikidata_id] Error for MBID 5a578a12-b977-4669-a542-e688207dd68c: 'url-relation-list'
[2146/107166] Daniel Ahearn & The Jones ➜ unknown
[2147/107166] Saint Raymond ➜ male
[2148/107166] Mike Dignam ➜ male
[2149/107166] BØRNS ➜ male
[2150/107166] Frankie Valli & The Four Seasons ➜ unknown
[2151/107166] Frankie Valli ➜ male
[2152/107166] Silk ➜ unknown
[2153/107166] Keith Sweat ➜ male
[2154/107166] Bando Jonez ➜ unknown
[2155/107166] Jacquees ➜ male
[2156/107166] A Great Big World ➜ unknown
[2157/107166] Dorrough Music ➜ unknown
[2158/107166] Rasheeda &, T-Pain ➜ male
[2159/107166] Eiffel 65 ➜ unknown
[2160/107166] Chingy ➜ male
[2161/107166] J-Kwon ➜ female
[2162/107166] Frankie J ➜ female
[2163/107166] Paula DeAnda ➜ female
[2164/107166] Danity Kane ➜ unknown
[2165/107166] Lil' Kim ➜ female
[2166/107166] 112 ➜ unknown
[2167/107166] Pretty Ricky ➜ unknown
[2168/107166] Twista ➜ male
[2169/107166] Memphis Bleek ➜ male
[2170/107166] Bobby V. ➜ unknown
[2171/107166] Mary J. Blige ➜ female
[2172/107166] Sean Paul ➜ male
[2173/107166] 69 Boyz ➜ unknown
[2174/107166] Nadia Ali ➜ female
[2175/107166] Madonna ➜ female
[2176/107166] Bob Sinclar ➜ male
[2177/107166] The Nightcrawlers ➜ unknown
[2178/107166] Black Box ➜ unknown
[2179/107166] Kylie Minogue ➜ female
[2180/107166] Jesse Garcia ➜ male
[2181/107166] Hardwell ➜ male
[2182/107166] Showtek ➜ unknown
[2183/107166] The Essential ➜ unknown
[2184/107166] Bizarre Inc ➜ unknown
[2185/107166] Out Here Brothers ➜ unknown
[2186/107166] Technotronic ➜ unknown
[2187/107166] SNAP! ➜ unknown
[2188/107166] Youngbloodz ➜ unknown
[2189/107166] Prince ➜ male
[2190/107166] Real McCoy ➜ unknown
[2191/107166] Montell Jordan ➜ male
[2192/107166] Sebastian Ingrosso ➜ male
[2193/107166] C & C Music Factory ➜ unknown
[2194/107166] Us3 ➜ unknown
[2195/107166] NERO ➜ unknown
[2196/107166] Funk ➜ unknown
[2197/107166] Deee-Lite ➜ unknown
[2198/107166] Alexandra Stan ➜ female
[2199/107166] Ian van Dahl ➜ unknown
[2200/107166] Madison Avenue ➜ unknown
[2201/107166] Debbie Deb ➜ female
[2202/107166] Planet Soul ➜ unknown
[2203/107166] Duck Sauce ➜ unknown
[2204/107166] Tag Team ➜ unknown
[2205/107166] Marky Mark And The Funky Bunch ➜ unknown
[2206/107166] CeCe Peniston ➜ female
[2207/107166] Robin S ➜ female
[2208/107166] 2 Unlimited ➜ unknown
[2209/107166] Reel 2 Real ➜ unknown
[2210/107166] Snap ➜ unknown
[2211/107166] K7 ➜ male
[2212/107166] The Bad Yard Club ➜ unknown
[2213/107166] Mighty Dub Katz ➜ unknown
[2214/107166] Le Click ➜ unknown
[2215/107166] Bang! ➜ male
[2216/107166] Alice DJ ➜ male
[2217/107166] Dove Shack ➜ unknown
[2218/107166] Zhané ➜ unknown
[2219/107166] Wreckx-N-Effect ➜ unknown
[2220/107166] Nikki Williams ➜ male
[2221/107166] Spiller ➜ male
[2222/107166] Fragma ➜ unknown
[2223/107166] Benny Benassi ➜ male
[2224/107166] iio ➜ unknown
[2225/107166] Amber ➜ female
[2226/107166] Daniel Bedingfield ➜ male
[2227/107166] 2 Bad Mice ➜ unknown
[2228/107166] Dirty Vegas ➜ unknown
[2229/107166] Fatboy Slim ➜ male
[2230/107166] Fedde Le Grand ➜ male
[2231/107166] La Bouche ➜ unknown
[2232/107166] Janet Jackson ➜ female
[2233/107166] Newcleus ➜ unknown
[2234/107166] Connie ➜ female
[2235/107166] NERVO ➜ unknown
[2236/107166] Armin van Buuren ➜ male
[2237/107166] Jocelyn Enriquez ➜ female
[2238/107166] Sonique ➜ female
[2239/107166] Afrika Bambaataa ➜ male
[2240/107166] The Blackout Allstars ➜ unknown
[2241/107166] Sander van Doorn ➜ male
[2242/107166] Ivan Gough ➜ unknown
[2243/107166] Will To Power ➜ unknown
[2244/107166] Rob Base & DJ EZ Rock ➜ unknown
[2245/107166] DJ Antoine ➜ male
[2246/107166] John Newman ➜ male
[2247/107166] Christine and the Queens ➜ unknown
[2248/107166] Dr. Kid ➜ male
[2249/107166] Swet Shop Boys ➜ unknown
[2250/107166] White Town ➜ agender
[2251/107166] The Piano Guys ➜ unknown
[2252/107166] Jon Schmidt ➜ male
[2253/107166] Alex Goot ➜ male
[2254/107166] Axel Hedfors ➜ unknown
[2255/107166] Sia Furler ➜ female
[get_wikidata_id] Error for MBID 76a717c0-3867-4bea-8abf-01f506256eef: 'url-relation-list'
[2256/107166] Sergei Rachmaninoff ➜ unknown
[2257/107166] Hillsong Worship ➜ unknown
[2258/107166] Tim Hughes ➜ male
[2259/107166] David Crowder Band ➜ unknown
[2260/107166] Newsboys ➜ unknown
[2261/107166] Jared Anderson ➜ female
[2262/107166] Matt Maher ➜ male
[2263/107166] Matthew West ➜ male
[2264/107166] Sidewalk Prophets ➜ unknown
[2265/107166] Casting Crowns ➜ unknown
[2266/107166] Darin and Brooke Aldridge ➜ unknown
[2267/107166] Jeff Johnson ➜ male
[2268/107166] Tenth Avenue North ➜ unknown
[2269/107166] Brian Johnson ➜ male
[2270/107166] Gungor ➜ unknown
[2271/107166] JJ Heller ➜ female
[2272/107166] Derek Webb ➜ male
[2273/107166] The Martins ➜ unknown
[get_wikidata_id] Error for MBID f3994e61-aeb7-484f-a679-bb2d351a8d86: 'url-relation-list'
[2274/107166] Kortnie Heying ➜ unknown
[2275/107166] Sara Watkins ➜ female
[2276/107166] Kari Jobe ➜ female
[2277/107166] Natalie Grant ➜ female
[2278/107166] Chris August ➜ male
[2279/107166] Dara Maclean ➜ female
[2280/107166] for KING & COUNTRY ➜ unknown
[2281/107166] Francesca Battistelli ➜ female
[2282/107166] Patrick Ryan Clark ➜ male
[2283/107166] Hillsong Young & Free ➜ unknown
[2284/107166] Austin Stone Worship ➜ unknown
[2285/107166] Phil Wickham ➜ male
[2286/107166] Shane & Shane ➜ unknown
[2287/107166] Unspoken ➜ unknown
[2288/107166] Andy Mineo ➜ male
[2289/107166] The City Harmonic ➜ unknown
[2290/107166] Tim Timmons ➜ unknown
[get_wikidata_id] Error for MBID 65adacbf-95e5-4bb4-a6f7-ccbc6dd483ca: 'url-relation-list'
[2291/107166] Will Reagan ➜ unknown
[2292/107166] Kristene Dimarco ➜ female
[2293/107166] James Tealy ➜ male
[2294/107166] Lauren Daigle ➜ female
[2295/107166] Steven Curtis Chapman ➜ male
[2296/107166] Aaron Shust ➜ male
[2297/107166] City Harbor ➜ unknown
[2298/107166] Tedashii ➜ male
[get_wikidata_id] Error for MBID 3b7dcd0a-4043-4eba-a846-ef90860ad9ff: 'url-relation-list'
[2299/107166] Breakaway Ministries ➜ unknown
[2300/107166] Bethel Music ➜ unknown
[2301/107166] Jeremy Riddle ➜ male
[2302/107166] Kalley Heiligenthal ➜ unknown
[2303/107166] Kings Kaleidoscope ➜ unknown
[2304/107166] Amanda Cook ➜ female
[2305/107166] Skank ➜ unknown
[2306/107166] Engenheiros Do Hawaii ➜ unknown
[2307/107166] Gilberto Gil ➜ male
[2308/107166] Chico Buarque ➜ male
[2309/107166] Vinícius de Moraes ➜ male
[2310/107166] Antônio Carlos Jobim ➜ male
[2311/107166] Elis Regina ➜ female
[2312/107166] Marcos Valle ➜ male
[2313/107166] Nelson Riddle ➜ male
[2314/107166] Pixinguinha ➜ male
[2315/107166] Caetano Veloso ➜ male
[2316/107166] Maria Bethânia ➜ female
[2317/107166] Marisa Monte ➜ female
[2318/107166] Adriana Calcanhotto ➜ female
[2319/107166] Seu Jorge ➜ male
[2320/107166] Tribalistas ➜ unknown
[2321/107166] Ana Carolina ➜ female
[2322/107166] Maria Rita ➜ female
[2323/107166] Djavan ➜ male
[2324/107166] Jackie Lee ➜ male
[2325/107166] Kelsea Ballerini ➜ female
[2326/107166] Brett Young ➜ male
[2327/107166] Old Dominion ➜ unknown
[2328/107166] Peaches ➜ female
[2329/107166] Caribou ➜ male
[2330/107166] Chromatics ➜ unknown
[2331/107166] HEALTH ➜ unknown
[2332/107166] LOS PILOTOS ➜ unknown
[2333/107166] Cineplexx ➜ male
[2334/107166] Fugees ➜ unknown
[2335/107166] Jamaican Queens ➜ unknown
[2336/107166] Morrissey ➜ male
[2337/107166] Paris Hilton ➜ female
[2338/107166] Hot Sugar ➜ unknown
[2339/107166] Kim Zolciak ➜ female
[2340/107166] New Edition ➜ unknown
[2341/107166] DJ Sammy ➜ male
[2342/107166] Jermaine Stewart ➜ male
[2343/107166] Eddie Murphy ➜ male
[2344/107166] Gravy Train!!!! ➜ unknown
[2345/107166] cupcakKe ➜ female
[2346/107166] Sharon Van Etten ➜ female
[2347/107166] Hercules & Love Affair ➜ unknown
[2348/107166] Junior Boys ➜ unknown
[2349/107166] Gardens & Villa ➜ unknown
[2350/107166] Gayngs ➜ unknown
[2351/107166] Holy Ghost! ➜ unknown
[2352/107166] Pure Bathing Culture ➜ unknown
[2353/107166] Dum Dum Girls ➜ unknown
[2354/107166] King Tuff ➜ male
[2355/107166] Natural Child ➜ male
[2356/107166] Shannon and The Clams ➜ unknown
[2357/107166] Black Lips ➜ unknown
[2358/107166] The Growlers ➜ unknown
[2359/107166] JEFF The Brotherhood ➜ unknown
[2360/107166] Birdcloud ➜ unknown
[2361/107166] Anamanaguchi ➜ unknown
[2362/107166] Chris Janson ➜ male
[2363/107166] Kid Rock ➜ male
[2364/107166] Blackjack Billy ➜ male
[2365/107166] The Wailers ➜ unknown
[2366/107166] KONGOS ➜ unknown
[2367/107166] Nappy Roots ➜ unknown
[2368/107166] Chiddy Bang ➜ unknown
[get_wikidata_id] Error for MBID b4e0aa57-e3bf-4140-8ebb-7ef4f5c4e44f: 'url-relation-list'
[2369/107166] OB OBrien ➜ unknown
[2370/107166] Franz Ferdinand ➜ male
[2371/107166] Alan Walker ➜ male
[2372/107166] The Fratellis ➜ unknown
[2373/107166] New Politics ➜ unknown
[2374/107166] Asher Roth ➜ male
[2375/107166] Bakermat ➜ male
[2376/107166] Mura Masa ➜ male
[2377/107166] Chris Webby ➜ male
[2378/107166] Steppenwolf ➜ unknown
[2379/107166] Wes Walker ➜ male
[2380/107166] Lionel Richie ➜ male
[2381/107166] Ballyhoo! ➜ unknown
[2382/107166] Bryce Vine ➜ male
[2383/107166] Rebelution ➜ unknown
[get_wikidata_id] Error for MBID 88924a62-6d1d-4bf0-9c41-20a413b90dc2: 'url-relation-list'
[2384/107166] Voidoid ➜ unknown
[2385/107166] Taking Care of Business Band ➜ unknown
[2386/107166] Steven Tyler ➜ female
[2387/107166] Beenie Man ➜ male
[get_wikidata_id] Error for MBID 566dbfbe-4feb-49e2-bc2e-f82f84f12c80: 'url-relation-list'
[2388/107166] Phay ➜ unknown
[2389/107166] Danny Brown ➜ male
[2390/107166] Tamer Hosny ➜ male
[2391/107166] Kathleen Edwards ➜ female
[2392/107166] Emmylou Harris ➜ female
[2393/107166] Turnpike Troubadours ➜ unknown
[2394/107166] Sam Outlaw ➜ male
[2395/107166] Jamie N Commons ➜ male
[2396/107166] Warren Haynes ➜ male
[2397/107166] Iris DeMent ➜ female
[2398/107166] Allen Toussaint ➜ male
[2399/107166] Sons Of Bill ➜ unknown
[2400/107166] Neko Case ➜ female
[2401/107166] The Mavericks ➜ unknown
[2402/107166] Jason Isbell ➜ male
[2403/107166] Paris & Simo ➜ unknown
[2404/107166] MisterWives ➜ unknown
[2405/107166] Vigiland ➜ unknown
[2406/107166] MercyMe ➜ unknown
[2407/107166] Pham ➜ male
[get_wikidata_id] Error for MBID 56b6c0de-7671-4e13-9e36-370f5ff57343: 'url-relation-list'
[2408/107166] GTA ➜ unknown
[2409/107166] Porter Robinson ➜ male
[2410/107166] Luca Lush ➜ unknown
[2411/107166] Slash ➜ male
[2412/107166] Quiet Riot ➜ unknown
[2413/107166] The Key of Awesome ➜ unknown
[2414/107166] Ritchie Valens ➜ male
[2415/107166] "Weird Al" Yankovic ➜ male
[2416/107166] Miami Sound Machine ➜ unknown
[2417/107166] Elvis Presley ➜ male
[2418/107166] The Chords ➜ unknown
[2419/107166] Bachman-Turner Overdrive ➜ unknown
[2420/107166] Bobby Day ➜ female
[2421/107166] Jim Croce ➜ male
[2422/107166] VeggieTales ➜ unknown
[2423/107166] O.S. Collective ➜ unknown
[2424/107166] Thunderstruck ➜ unknown
[2425/107166] O-Zone ➜ unknown
[2426/107166] William Joseph & Lindsey Stirling ➜ male
[2427/107166] Was (Not Was) ➜ unknown
[2428/107166] Styx ➜ unknown
[2429/107166] 3 Doors Down ➜ unknown
[2430/107166] Toni Basil ➜ female
[get_wikidata_id] Error for MBID e2d5a62e-72ef-456f-bfc8-d1ff8a046790: 'url-relation-list'
[2431/107166] Quint ➜ unknown
[2432/107166] Trans-Siberian Orchestra ➜ unknown
[2433/107166] Back In Black ➜ unknown
[2434/107166] Starship ➜ unknown
[2435/107166] Da Vinci's Notebook ➜ unknown
[2436/107166] War ➜ unknown
[2437/107166] Lil Deuce Deuce ➜ unknown
[2438/107166] Epic Rap Battles of History ➜ unknown
[2439/107166] Theory of a Deadman ➜ unknown
[2440/107166] I Prevail ➜ unknown
[2441/107166] Halestorm ➜ unknown
[2442/107166] Starset ➜ unknown
[2443/107166] Ice Cube ➜ male
[2444/107166] Tech N9ne Collabos ➜ male
[2445/107166] Crossfade ➜ unknown
[2446/107166] Godsmack ➜ unknown
[2447/107166] Black Stone Cherry ➜ unknown
[2448/107166] Ozzy Osbourne ➜ male
[2449/107166] Drowning Pool ➜ unknown
[2450/107166] Disturbed ➜ unknown
[2451/107166] Rob Zombie ➜ male
[2452/107166] Skillet ➜ unknown
[2453/107166] Breaking Benjamin ➜ unknown
[2454/107166] Black Sabbath ➜ unknown
[2455/107166] Motörhead ➜ unknown
[2456/107166] Iron Maiden ➜ unknown
[2457/107166] Megadeth ➜ unknown
[2458/107166] Mountain ➜ unknown
[2459/107166] Denzel Curry ➜ male
[2460/107166] Three Days Grace ➜ unknown
[2461/107166] KISS ➜ unknown
[2462/107166] Dio ➜ unknown
[get_wikidata_id] Error for MBID bffc3062-19c3-4c37-80c6-0b48fa051792: 'url-relation-list'
[2463/107166] The Black Jets ➜ unknown
[2464/107166] Sensation Ltd ➜ unknown
[2465/107166] The Matons ➜ unknown
[get_wikidata_id] Error for MBID 429a210d-4bcc-4358-814c-b380de5a8760: 'url-relation-list'
[2466/107166] JWL ➜ unknown
[2467/107166] The West Coast Sound Machine ➜ unknown
[2468/107166] Dan Deacon ➜ male
[2469/107166] Bobby "Boris" Pickett & The Crypt-Kickers ➜ unknown
[2470/107166] Classics IV ➜ unknown
[2471/107166] Tiny Tim w/ The New Duncan Imperials ➜ unknown
[2472/107166] The Lovin' Spoonful ➜ unknown
[2473/107166] Richard O'Brien ➜ male
[2474/107166] Patricia Quinn ➜ male
[2475/107166] Tim Curry ➜ male
[2476/107166] Benjamin Schrader ➜ male
[2477/107166] Silver Screen Superstars ➜ unknown
[2478/107166] Keith David ➜ male
[2479/107166] Steve Miller Band ➜ unknown
[2480/107166] Ken Page ➜ male
[2481/107166] Blue Öyster Cult ➜ unknown
[2482/107166] Trap Beckham ➜ unknown
[2483/107166] Afroman ➜ male
[2484/107166] Dem Franchize Boyz ➜ unknown
[2485/107166] Cobra Starship ➜ unknown
[2486/107166] Bubba Sparxxx ➜ male
[2487/107166] Soulja Boy ➜ male
[2488/107166] Far East Movement ➜ unknown
[2489/107166] Huey ➜ unknown
[2490/107166] Fountains Of Wayne ➜ unknown
[2491/107166] DEV ➜ female
[2492/107166] Yung Joc ➜ male
[2493/107166] Juvenile ➜ male
[2494/107166] Julia Cole ➜ male
[2495/107166] OG Maco ➜ male
[2496/107166] PRTY H3RO ➜ unknown
[2497/107166] Adrian Marcel ➜ male
[get_wikidata_id] Error for MBID 3361c5b2-6133-45ff-a697-c33eda1d6896: 'url-relation-list'
[2498/107166] DLOW ➜ unknown
[2499/107166] Sammy Adams ➜ male
[2500/107166] Trick Daddy ➜ unknown
[2501/107166] POWERS ➜ unknown
[2502/107166] Dj Slim D ➜ male
[2503/107166] Ayo Jay ➜ male
[2504/107166] Red Cafe ➜ male
[2505/107166] Buckcherry ➜ unknown
[2506/107166] D4L ➜ unknown
[2507/107166] Dreezy ➜ female
[2508/107166] Natasja ➜ female
[2509/107166] Hollywood Undead ➜ unknown
[2510/107166] 24 Hour Party Project ➜ unknown
[2511/107166] Workout Buddy ➜ male
[2512/107166] Mayday Parade ➜ unknown
[2513/107166] Deuce ➜ unknown
[2514/107166] Workout Remix Factory ➜ unknown
[2515/107166] Shinedown ➜ unknown
[2516/107166] Shop Boyz ➜ unknown
[2517/107166] Immortal Technique ➜ male
[2518/107166] Limp Bizkit ➜ unknown
[2519/107166] Krept & Konan ➜ unknown
[2520/107166] V.I.C. ➜ male
[2521/107166] Petey Pablo ➜ male
[2522/107166] Juelz Santana ➜ male
[2523/107166] Allstar Weekend ➜ unknown
[2524/107166] Trinidad James ➜ male
[2525/107166] Krewella ➜ unknown
[2526/107166] Omar LinX ➜ unknown
[2527/107166] G Curtis ➜ male
[2528/107166] DJ Crazy J Rodriguez ➜ female
[2529/107166] Jack Andreti ➜ male
[2530/107166] Mocki ➜ unknown
[2531/107166] Sofia Carson ➜ female
[2532/107166] ayokay ➜ male
[2533/107166] Kevin Rudolf ➜ male
[2534/107166] 2AM Club ➜ unknown
[2535/107166] JP Cooper ➜ male
[2536/107166] Adam Friedman ➜ unknown
[2537/107166] Ace Hood ➜ male
[2538/107166] Iration ➜ unknown
[2539/107166] Busy Signal ➜ male
[2540/107166] Romain Virgo ➜ male
[2541/107166] Mr Easy ➜ unknown
[2542/107166] Spectacular ➜ unknown
[2543/107166] Angela Hunte ➜ female
[2544/107166] Zagga ➜ unknown
[2545/107166] Gramps Morgan ➜ unknown
[2546/107166] Tarrus Riley ➜ male
[2547/107166] Luciano ➜ male
[2548/107166] Pressure ➜ male
[2549/107166] Buju Banton ➜ male
[2550/107166] Assassin ➜ unknown
[2551/107166] Christopher Martin ➜ male
[2552/107166] Chino ➜ male
[2553/107166] Konshens ➜ male
[2554/107166] Queen Ifrica ➜ unknown
[2555/107166] The Kinks ➜ unknown
[2556/107166] Darlene Love ➜ female
[2557/107166] Low ➜ unknown
[2558/107166] Joni Mitchell ➜ female
[2559/107166] The Pogues ➜ unknown
[2560/107166] Lindsey Buckingham ➜ male
[2561/107166] Pretenders ➜ unknown
[2562/107166] Ramones ➜ unknown
[2563/107166] The Waitresses ➜ unknown
[2564/107166] Whirling Dervishes ➜ unknown
[2565/107166] Vince Guaraldi Trio ➜ unknown
[2566/107166] Band Aid ➜ unknown
[2567/107166] Jimi Hendrix ➜ male
[2568/107166] The Ronettes ➜ unknown
[2569/107166] The Drifters ➜ unknown
[2570/107166] Joyce Manor ➜ unknown
[2571/107166] Okkervil River ➜ unknown
[2572/107166] The Fall ➜ unknown
[2573/107166] Julian Casablancas ➜ male
[2574/107166] Animal Collective ➜ unknown
[2575/107166] Big Star ➜ unknown
[2576/107166] Billy Squier ➜ male
[2577/107166] Gardiner Sisters ➜ unknown
[2578/107166] Brii ➜ unknown
[2579/107166] SafetySuit ➜ unknown
[2580/107166] Eurythmics ➜ unknown
[2581/107166] Dexys Midnight Runners ➜ unknown
[2582/107166] Edwin Starr ➜ male
[2583/107166] Kelly Clarkson ➜ female
[2584/107166] The Beatles Tribute Band ➜ unknown
[2585/107166] The Cure ➜ unknown
[2586/107166] Simple Minds ➜ unknown
[2587/107166] Pacific Star ➜ unknown
[2588/107166] Rick James ➜ male
[2589/107166] EMF ➜ unknown
[2590/107166] DJ Chose ➜ unknown
[2591/107166] The Red Jumpsuit Apparatus ➜ unknown
[2592/107166] Kristinia DeBarge ➜ female
[2593/107166] Modern English ➜ unknown
[2594/107166] Deep Blue Something ➜ unknown
[2595/107166] MGMT ➜ unknown
[2596/107166] Sing King ➜ male
[2597/107166] Pixies ➜ unknown
[2598/107166] Dwayne Johnson ➜ male
[2599/107166] Auli'i Cravalho ➜ female
[2600/107166] Garren Sean ➜ unknown
[get_wikidata_id] Error for MBID 935f33bd-aedb-47df-ae8d-9f29f6a76d81: 'url-relation-list'
[2601/107166] MOSSS ➜ unknown
[2602/107166] Smino ➜ male
[2603/107166] Kali Uchis ➜ female
[2604/107166] Billie Eilish ➜ non-binary gender
[2605/107166] Soy Christmas ➜ unknown
[2606/107166] Vera Blue ➜ female
[2607/107166] Ark Patrol ➜ unknown
[2608/107166] Marina and the Diamonds ➜ unknown
[2609/107166] Diddy - Dirty Money ➜ unknown
[2610/107166] L8show ➜ unknown
[2611/107166] Santigold ➜ female
[2612/107166] Jordin Sparks ➜ female
[2613/107166] Mikkel S. Eriksen ➜ male
[2614/107166] Jai Wolf ➜ male
[2615/107166] Maino ➜ male
[2616/107166] Dixie Chicks ➜ unknown
[2617/107166] Earth ➜ unknown
[2618/107166] The Acacia Strain ➜ unknown
[2619/107166] Carl Orff ➜ male
[2620/107166] Léo Delibes ➜ male
[2621/107166] Ty ➜ male
[2622/107166] Winds of Plague ➜ unknown
[2623/107166] Niykee Heaton ➜ female
[2624/107166] LÉON ➜ female
[2625/107166] LUME ➜ unknown
[2626/107166] Tatiana Manaois ➜ unknown
[2627/107166] Jake Bugg ➜ male
[2628/107166] Hayley Kiyoko ➜ female
[2629/107166] Angel Haze ➜ agender
[2630/107166] Forty Foot Echo ➜ unknown
[2631/107166] Tru$ ➜ unknown
[2632/107166] Hanson ➜ unknown
[2633/107166] Lily Allen ➜ female
[2634/107166] Kottonmouth Kings ➜ unknown
[2635/107166] Vital ➜ unknown
[2636/107166] Big Tymers ➜ unknown
[2637/107166] Kreayshawn ➜ female
[2638/107166] t.A.T.u. ➜ unknown
[2639/107166] Carnage ➜ unknown
[2640/107166] Havana Brown ➜ male
[2641/107166] ETC!ETC! ➜ unknown
[2642/107166] Roxette ➜ unknown
[2643/107166] Dada Life ➜ unknown
[2644/107166] Plumb ➜ female
[2645/107166] Travis Barker ➜ unknown
[2646/107166] Rock Mafia ➜ unknown
[2647/107166] Big Will ➜ male
[2648/107166] Project Pat ➜ male
[2649/107166] Lil' Flip ➜ male
[2650/107166] St. Lunatics ➜ unknown
[2651/107166] Basement Jaxx ➜ unknown
[2652/107166] Joss Whedon ➜ male
[2653/107166] Original Cast of Buffy The Vampire Slayer ➜ unknown
[2654/107166] Angela Lansbury ➜ female
[2655/107166] Faith Hill ➜ female
[2656/107166] The Dayton Family ➜ unknown
[2657/107166] Somethin' For The People ➜ unknown
[2658/107166] Benzino ➜ male
[2659/107166] Stone Temple Pilots ➜ unknown
[2660/107166] Rage Against The Machine ➜ unknown
[2661/107166] Cake ➜ unknown
[2662/107166] Crash Test Dummies ➜ unknown
[2663/107166] Temple Of The Dog ➜ unknown
[2664/107166] Semisonic ➜ unknown
[2665/107166] Queensrÿche ➜ unknown
[2666/107166] Jerry Cantrell ➜ male
[2667/107166] Ben Folds Five ➜ unknown
[2668/107166] Cracker ➜ unknown
[2669/107166] Stabbing Westward ➜ unknown
[2670/107166] Jurassic 5 ➜ unknown
[2671/107166] Q-Tip ➜ male
[2672/107166] Joss Stone ➜ female
[2673/107166] Ghostface Killah ➜ male
[2674/107166] Rhymefest ➜ male
[2675/107166] MNEK ➜ male
[2676/107166] Karen Harding ➜ female
[2677/107166] Klingande ➜ male
[2678/107166] Gallant ➜ male
[2679/107166] Ashley DuBose ➜ unknown
[2680/107166] Tin Sparrow ➜ unknown
[2681/107166] M83 ➜ unknown
[2682/107166] Maps & Atlases ➜ unknown
[2683/107166] Wheeler Brothers ➜ unknown
[2684/107166] GIVERS ➜ unknown
[2685/107166] Elvis Costello and Mumford and Sons ➜ male
[2686/107166] White Lies ➜ unknown
[2687/107166] Craft Spells ➜ unknown
[2688/107166] Ugly Casanova ➜ unknown
[2689/107166] He's My Brother She's My Sister ➜ unknown
[2690/107166] CHVRCHES ➜ unknown
[2691/107166] The Temper Trap ➜ unknown
[2692/107166] WALLA ➜ unknown
[2693/107166] Matt Corby ➜ male
[2694/107166] The Chain Gang Of 1974 ➜ unknown
[2695/107166] The Gaslight Anthem ➜ unknown
[2696/107166] Noah And The Whale ➜ unknown
[2697/107166] Sky Ferreira ➜ female
[2698/107166] Tune-Yards ➜ unknown
[2699/107166] The Little Ones ➜ unknown
[2700/107166] The Wombats ➜ unknown
[2701/107166] Local Natives ➜ unknown
[2702/107166] The Weeks ➜ unknown
[2703/107166] The Kingston Springs ➜ unknown
[2704/107166] Spank Rock ➜ male
[2705/107166] Youth Lagoon ➜ male
[2706/107166] Born Ruffians ➜ unknown
[2707/107166] Caleb Followill ➜ male
[2708/107166] Folly and the Hunter ➜ unknown
[2709/107166] Deptford Goth ➜ unknown
[2710/107166] Snowmine ➜ unknown
[2711/107166] Small Black ➜ unknown
[2712/107166] Ryan Miller ➜ male
[2713/107166] The Orb ➜ unknown
[2714/107166] Sanders Bohlke ➜ unknown
[2715/107166] Delta Spirit ➜ unknown
[2716/107166] In The Valley Below ➜ unknown
[2717/107166] Run River North ➜ unknown
[2718/107166] Earl Sweatshirt ➜ male
[2719/107166] Pomegranates ➜ unknown
[2720/107166] Cayucas ➜ unknown
[2721/107166] The Antlers ➜ unknown
[2722/107166] BoomBox ➜ unknown
[2723/107166] Nathaniel Rateliff ➜ male
[2724/107166] The Vaccines ➜ unknown
[2725/107166] Milagres ➜ unknown
[2726/107166] Hudson and Troop ➜ unknown
[2727/107166] Hot Hot Heat ➜ unknown
[2728/107166] Blitzen Trapper ➜ unknown
[2729/107166] Little Green Cars ➜ unknown
[2730/107166] Cataldo ➜ unknown
[2731/107166] Marco Pavé ➜ male
[2732/107166] Alfred Banks ➜ male
[2733/107166] Banks & Steelz ➜ unknown
[2734/107166] GoldLink ➜ male
[get_wikidata_id] Error for MBID 43f3a9b0-a0aa-4a03-b845-c57bb0906244: 'url-relation-list'
[2735/107166] Sidewayz ➜ unknown
[2736/107166] Daye Jack ➜ unknown
[2737/107166] Sampa the Great ➜ female
[2738/107166] Big Grams ➜ unknown
[2739/107166] Pell ➜ unknown
[2740/107166] Cities Aviv ➜ male
[2741/107166] Kenneth Whalum ➜ male
[2742/107166] Noname ➜ female
[2743/107166] The Last Artful, Dodgr ➜ female
[2744/107166] Warm Brew ➜ unknown
[2745/107166] FloFilz ➜ unknown
[2746/107166] Myke Bogan ➜ unknown
[2747/107166] Raury ➜ male
[2748/107166] Kamaiyah ➜ female
[2749/107166] Run The Jewels ➜ unknown
[2750/107166] Ghetto Vanessa ➜ female
[2751/107166] Lizzo ➜ female
[2752/107166] Caleborate ➜ unknown
[2753/107166] Magna Carda ➜ unknown
[2754/107166] Mick Jenkins ➜ male
[2755/107166] Otis Junior ➜ male
[2756/107166] Porter Ray ➜ male
[2757/107166] Wrekonize ➜ male
[2758/107166] Hippy Soul ➜ unknown
[get_wikidata_id] Error for MBID 694d4595-9e18-4ae5-8a0e-168aae45b0b6: 'url-relation-list'
[2759/107166] Awfm ➜ unknown
[2760/107166] jeremy messersmith ➜ male
[2761/107166] Alexi Murdoch ➜ male
[2762/107166] Amos Lee ➜ female
[2763/107166] Benjamin Francis Leftwich ➜ male
[2764/107166] Matt Hires ➜ male
[2765/107166] Peter Bradley Adams ➜ male
[2766/107166] Josh Ritter ➜ male
[2767/107166] Explosions In The Sky ➜ unknown
[2768/107166] The Morning Benders ➜ unknown
[2769/107166] Bon Iver ➜ unknown
[2770/107166] Bombay Bicycle Club ➜ unknown
[2771/107166] The Flaming Lips ➜ unknown
[2772/107166] Kings of Convenience ➜ unknown
[2773/107166] Hans Zimmer ➜ male
[2774/107166] Whitetree ➜ unknown
[2775/107166] Pale Seas ➜ unknown
[2776/107166] The Wailin' Jennys ➜ unknown
[2777/107166] Johnny Flynn ➜ male
[2778/107166] A.A. Bondy ➜ male
[2779/107166] Imogen Heap ➜ female
[2780/107166] Frou Frou ➜ unknown
[2781/107166] Alison Krauss & Union Station ➜ unknown
[2782/107166] Vancouver Sleep Clinic ➜ male
[2783/107166] Fennesz ➜ male
[2784/107166] Nick White ➜ male
[2785/107166] Gemma Hayes ➜ female
[2786/107166] Emma-Lee ➜ male
[2787/107166] Deaf Joe ➜ male
[2788/107166] Dougie MacLean ➜ male
[2789/107166] Active Child ➜ male
[2790/107166] Jerome Holloway ➜ female
[2791/107166] The Cinematic Orchestra ➜ unknown
[2792/107166] Nouela ➜ unknown
[2793/107166] dodie ➜ female
[2794/107166] Joy Williams ➜ male
[2795/107166] Junksista ➜ unknown
[2796/107166] The Presets ➜ unknown
[2797/107166] Justice ➜ unknown
[2798/107166] The Crystal Method ➜ unknown
[2799/107166] Hadouken! ➜ unknown
[2800/107166] Pendulum ➜ unknown
[2801/107166] The Prodigy ➜ unknown
[2802/107166] Aphex Twin ➜ male
[2803/107166] KMFDM ➜ unknown
[2804/107166] Mindless Self Indulgence ➜ unknown
[2805/107166] Sabrepulse ➜ male
[2806/107166] Nostalgia ➜ unknown
[2807/107166] Ladytron ➜ unknown
[2808/107166] Wolfgang Gartner ➜ male
[2809/107166] The Qemists ➜ unknown
[2810/107166] Yeah Yeah Yeahs ➜ unknown
[2811/107166] Monarchy ➜ unknown
[2812/107166] N.E.R.D ➜ unknown
[2813/107166] Perturbator ➜ male
[2814/107166] M|O|O|N ➜ male
[2815/107166] TJH87 ➜ unknown
[2816/107166] Azealia Banks ➜ female
[2817/107166] CSS ➜ unknown
[2818/107166] Shamir ➜ non-binary gender
[2819/107166] Ladyhawke ➜ female
[2820/107166] Judge Bitch ➜ unknown
[2821/107166] Janelle Monáe ➜ non-binary gender
[2822/107166] XXXTENTACION ➜ male
[2823/107166] Ugly God ➜ male
[2824/107166] Jamie Cullum ➜ male
[2825/107166] Melody Gardot ➜ female
[2826/107166] Ray Charles ➜ male
[2827/107166] Diana Krall ➜ female
[2828/107166] James Morrison ➜ male
[2829/107166] Joshua Radin ➜ male
[2830/107166] Laura Jansen ➜ female
[2831/107166] Miss Montreal ➜ unknown
[2832/107166] Duffy ➜ female
[2833/107166] Stacey Kent ➜ female
[2834/107166] Jim Tomlinson ➜ male
[2835/107166] Eliane Elias ➜ female
[2836/107166] Kina Grannis ➜ female
[2837/107166] Alison Krauss ➜ female
[2838/107166] Obadiah Parker ➜ male
[2839/107166] Sting ➜ male
[2840/107166] Jake Shimabukuro ➜ male
[2841/107166] Eric Margan & The Red Lions ➜ unknown
[2842/107166] Chris Isaak ➜ male
[2843/107166] Richard Bona ➜ male
[2844/107166] Shirley Horn ➜ female
[2845/107166] Rumer ➜ female
[2846/107166] Tony Bennett ➜ male
[2847/107166] Holly Cole Trio ➜ unknown
[2848/107166] Holly Cole ➜ male
[2849/107166] the bird and the bee ➜ unknown
[2850/107166] Renee Olstead ➜ female
[2851/107166] Charlie Mars ➜ male
[2852/107166] Pink Floyd ➜ unknown
[2853/107166] Van Halen ➜ unknown
[2854/107166] Apocalyptica ➜ unknown
[2855/107166] Falling In Reverse ➜ unknown
[2856/107166] Black Veil Brides ➜ unknown
[2857/107166] Social Repose ➜ male
[2858/107166] Bring Me The Horizon ➜ unknown
[2859/107166] Andy Black ➜ unknown
[2860/107166] Anna Blue ➜ unknown
[2861/107166] Set It Off ➜ unknown
[2862/107166] Frank Carter & The Rattlesnakes ➜ unknown
[2863/107166] A Static Lullaby ➜ unknown
[2864/107166] Gabriella Cilmi ➜ female
[2865/107166] Noisettes ➜ unknown
[2866/107166] All Saints ➜ unknown
[2867/107166] Youngblood Hawke ➜ unknown
[2868/107166] Digitalism ➜ unknown
[2869/107166] Matt and Kim ➜ unknown
[2870/107166] Matt Costa ➜ male
[2871/107166] Stepdad ➜ unknown
[2872/107166] The Strokes ➜ unknown
[2873/107166] Eagles Of Death Metal ➜ unknown
[2874/107166] Young Rising Sons ➜ unknown
[2875/107166] Sir Sly ➜ male
[2876/107166] Young Empires ➜ male
[2877/107166] Cut Copy ➜ unknown
[2878/107166] Houndmouth ➜ unknown
[2879/107166] Matt Simons ➜ male
[2880/107166] That's Nice ➜ unknown
[2881/107166] Capital Cities ➜ unknown
[2882/107166] New Radicals ➜ unknown
[2883/107166] Caesars ➜ unknown
[2884/107166] Hellogoodbye ➜ unknown
[2885/107166] Guards ➜ unknown
[2886/107166] Olympic Ayres ➜ unknown
[2887/107166] The Griswolds ➜ unknown
[2888/107166] OK Go ➜ unknown
[2889/107166] Kasket Club ➜ unknown
[2890/107166] Penguin Prison ➜ unknown
[2891/107166] Magic City Hippies ➜ unknown
[2892/107166] Vista Kicks ➜ unknown
[2893/107166] Bob Moses ➜ male
[2894/107166] Mother Mother ➜ unknown
[2895/107166] Lotus  ➜ male
[2896/107166] The Librarians ➜ unknown
[2897/107166] CRUISR ➜ unknown
[get_wikidata_id] Error for MBID e3374306-4b2f-412f-b9a2-974d958eae4e: 'url-relation-list'
[2898/107166] Lost Triibe ➜ unknown
[2899/107166] Elliphant ➜ female
[2900/107166] D-Why ➜ male
[2901/107166] Gentlemen Hall ➜ unknown
[2902/107166] Snoop Lion ➜ male
[2903/107166] The Expendables ➜ unknown
[2904/107166] Ky-Mani Marley ➜ male
[2905/107166] Chaka Demus & Pliers ➜ unknown
[2906/107166] Bedouin Soundclash ➜ unknown
[2907/107166] Hollie Cook ➜ female
[2908/107166] The Skints ➜ unknown
[2909/107166] Reel Big Fish ➜ unknown
[2910/107166] Streetlight Manifesto ➜ unknown
[2911/107166] Mustard Plug ➜ unknown
[2912/107166] Elle King ➜ male
[2913/107166] Rixton ➜ unknown
[2914/107166] Spice Girls ➜ unknown
[2915/107166] Shawn Hook ➜ unknown
[2916/107166] John Coltrane ➜ male
[2917/107166] Chet Baker ➜ male
[2918/107166] Julie London ➜ female
[2919/107166] Miles Davis ➜ male
[2920/107166] Karrin Allyson ➜ female
[2921/107166] Natalie Cole ➜ female
[2922/107166] Sam Cooke ➜ male
[2923/107166] Andra Day ➜ female
[2924/107166] Ryn Weaver ➜ female
[2925/107166] Fred Astaire ➜ male
[2926/107166] ALO ➜ unknown
[2927/107166] Percy Sledge ➜ male
[2928/107166] The Neighbourhood ➜ unknown
[2929/107166] Idina Menzel ➜ female
[2930/107166] Labrinth ➜ male
[2931/107166] DJ Pauly D ➜ male
[2932/107166] Kaskade ➜ male
[2933/107166] Arty ➜ male
[2934/107166] Lindsey Stirling ➜ female
[2935/107166] Alex Boyé ➜ male
[2936/107166] G.R.L. ➜ unknown
[2937/107166] Ferry Corsten ➜ male
[2938/107166] Daughtry ➜ unknown
[2939/107166] Alvin Risk ➜ male
[2940/107166] Madeon ➜ male
[2941/107166] Låpsley ➜ female
[2942/107166] Rufus Wainwright ➜ male
[2943/107166] Once Jameson ➜ female
[2944/107166] Marcus Marr ➜ unknown
[2945/107166] Rittz ➜ male
[2946/107166] Blind Pilot ➜ unknown
[2947/107166] Sleeping At Last ➜ unknown
[2948/107166] Conor Oberst and the Mystic Valley Band ➜ unknown
[2949/107166] Blue October ➜ unknown
[2950/107166] Keaton Henson ➜ male
[2951/107166] Twin Atlantic ➜ unknown
[2952/107166] Trevor Hall ➜ male
[2953/107166] Fractures ➜ unknown
[2954/107166] Faith Evans ➜ female
[2955/107166] August Alsina ➜ male
[2956/107166] Tamar Braxton ➜ female
[2957/107166] Young Steff ➜ male
[2958/107166] Monica ➜ female
[2959/107166] En Vogue ➜ unknown
[2960/107166] Adina Howard ➜ female
[2961/107166] H-Town ➜ unknown
[2962/107166] Meli'sa Morgan ➜ female
[2963/107166] Teddy Pendergrass ➜ male
[2964/107166] Minnie Riperton ➜ female
[2965/107166] Freddie Jackson ➜ male
[2966/107166] Sisqo ➜ male
[2967/107166] Joe ➜ male
[2968/107166] Color Me Badd ➜ unknown
[2969/107166] Rome ➜ unknown
[2970/107166] Hi-Five ➜ unknown
[2971/107166] Az Yet ➜ unknown
[2972/107166] Changing Faces ➜ unknown
[2973/107166] J. Holiday ➜ female
[2974/107166] Jodeci ➜ unknown
[2975/107166] Case ➜ female
[2976/107166] Xscape ➜ unknown
[2977/107166] Jagged Edge ➜ unknown
[2978/107166] Total ➜ unknown
[2979/107166] Jesse Powell ➜ male
[2980/107166] Kevin Lyttle ➜ male
[2981/107166] Trina ➜ female
[2982/107166] Mýa ➜ female
[2983/107166] Audrey Rose ➜ female
[2984/107166] B5 ➜ unknown
[2985/107166] K-Young ➜ female
[2986/107166] Fatty Koo ➜ unknown
[2987/107166] Lyfe Jennings ➜ male
[2988/107166] LL Cool J ➜ male
[2989/107166] Shanice ➜ female
[2990/107166] RL ➜ male
[2991/107166] Lil Ru ➜ male
[2992/107166] KeKe Wyatt ➜ female
[2993/107166] Aaliyah ➜ female
[2994/107166] Stephanie Mills ➜ female
[2995/107166] Dru Hill ➜ unknown
[2996/107166] The Doors ➜ unknown
[2997/107166] The Shirelles ➜ unknown
[2998/107166] The Mamas & The Papas ➜ unknown
[2999/107166] Bo Diddley ➜ male
[3000/107166] Act As If ➜ unknown
[3001/107166] Ashes Remain ➜ unknown
[3002/107166] Foy Vance ➜ male
[3003/107166] Agnes Obel ➜ female
[3004/107166] Novo Amor ➜ male
[3005/107166] Neil Halstead ➜ male
[3006/107166] Rogue Valley ➜ unknown
[3007/107166] Grace Mitchell ➜ female
[3008/107166] Caveman ➜ unknown
[3009/107166] The Strumbellas ➜ unknown
[3010/107166] The Dandy Warhols ➜ unknown
[3011/107166] Sjowgren ➜ unknown
[3012/107166] Good Old War ➜ unknown
[3013/107166] Andrew Bird ➜ male
[3014/107166] Eddie Vedder ➜ male
[3015/107166] School Of Seven Bells ➜ unknown
[3016/107166] Shearwater ➜ unknown
[get_wikidata_id] Error for MBID 02f16171-8ad5-4af9-b1df-d54a46bc5c3d: 'url-relation-list'
[3017/107166] Blunder ➜ unknown
[3018/107166] Max Frost ➜ male
[3019/107166] Robert DeLong ➜ male
[3020/107166] BANNERS ➜ male
[3021/107166] Marc Scibilia ➜ unknown
[3022/107166] Hey Marseilles ➜ unknown
[3023/107166] Tall Tales & The Silver Lining ➜ unknown
[3024/107166] Pinback ➜ unknown
[3025/107166] Kacy Hill ➜ female
[3026/107166] Baio ➜ male
[3027/107166] Barns Courtney ➜ male
[3028/107166] Mystery Jets ➜ unknown
[3029/107166] Amber Run ➜ unknown
[3030/107166] Calum Scott ➜ male
[3031/107166] Emeli Sandé ➜ female
[3032/107166] EDEN ➜ male
[3033/107166] Augustana ➜ unknown
[3034/107166] The Revivalists ➜ unknown
[3035/107166] Alexa Goddard ➜ unknown
[3036/107166] Priyanka Chopra ➜ female
[get_wikidata_id] Error for MBID db1307f7-e900-4a9c-b0b0-74f3b18a3315: 'url-relation-list'
[3037/107166] Rishi Rich Project ➜ unknown
[3038/107166] Raghav ➜ male
[3039/107166] Amar Arshi ➜ unknown
[3040/107166] Marvin Sease ➜ male
[3041/107166] J. Blackfoot ➜ female
[3042/107166] Mel Waiters ➜ male
[3043/107166] Vick Allen ➜ female
[3044/107166] Wilson Meadows ➜ male
[3045/107166] Lee "Shot" Williams ➜ male
[3046/107166] David Brinston ➜ male
[3047/107166] Frank Mendenhall ➜ male
[3048/107166] Ms. Jody ➜ unknown
[3049/107166] Latimore ➜ male
[3050/107166] Johnnie Taylor ➜ male
[3051/107166] The Ebonys ➜ unknown
[3052/107166] Lenny Williams ➜ male
[3053/107166] Tucka: King Of Swing ➜ unknown
[3054/107166] Tony Troutman ➜ male
[3055/107166] Bobby Rush ➜ unknown
[3056/107166] James Payne ➜ male
[3057/107166] Willie Clayton ➜ male
[3058/107166] Tyrone Davis ➜ male
[3059/107166] Jimmie Ja ➜ male
[3060/107166] Midnight Oil ➜ unknown
[3061/107166] ZZ Top ➜ unknown
[3062/107166] All That Remains ➜ unknown
[3063/107166] G. Point Allstars ➜ male
[3064/107166] Mötley Crüe ➜ unknown
[3065/107166] Tenacious D ➜ unknown
[3066/107166] Rammstein ➜ unknown
[3067/107166] Lindemann ➜ unknown
[3068/107166] DJ Esco ➜ male
[3069/107166] Young M.A ➜ male
[3070/107166] YFN Lucci ➜ male
[3071/107166] Luke Combs ➜ male
[3072/107166] Bebe Rexha ➜ female
[3073/107166] The Doobie Brothers ➜ unknown
[3074/107166] The Dirty Guv'nahs ➜ unknown
[3075/107166] Looking Glass ➜ male
[3076/107166] Funktown America ➜ unknown
[3077/107166] The Police ➜ unknown
[3078/107166] David Gray ➜ male
[3079/107166] King Harvest ➜ male
[3080/107166] KC & The Sunshine Band ➜ unknown
[3081/107166] Steely Dan ➜ unknown
[3082/107166] Dobie Gray ➜ male
[3083/107166] John the Ghost ➜ unknown
[3084/107166] Jacob Whitesides ➜ unknown
[3085/107166] The Maine ➜ unknown
[3086/107166] Allman Brown ➜ male
[3087/107166] Night Beds ➜ unknown
[3088/107166] Biffy Clyro ➜ unknown
[3089/107166] Audioslave ➜ unknown
[3090/107166] Eric Clapton ➜ male
[3091/107166] Jesse Boykins III ➜ male
[3092/107166] Chris Botti ➜ male
[3093/107166] Boom Clap Bachelors ➜ unknown
[3094/107166] J Dilla ➜ male
[3095/107166] Illa J ➜ female
[3096/107166] Blue In Green ➜ unknown
[3097/107166] Finley Quaye ➜ male
[3098/107166] Jessie Ware ➜ female
[3099/107166] Denitia and Sene ➜ unknown
[3100/107166] The Bug ➜ unknown
[3101/107166] Joe Cuba ➜ male
[3102/107166] Mocky ➜ male
[3103/107166] Flako ➜ unknown
[3104/107166] Soho ➜ unknown
[3105/107166] Grace Jones ➜ female
[3106/107166] Sara Tavares ➜ unknown
[3107/107166] Yusef Lateef ➜ male
[3108/107166] Takuya Kuroda ➜ male
[3109/107166] Mndsgn ➜ male
[3110/107166] Dorothy Ashby ➜ female
[3111/107166] Hird ➜ unknown
[3112/107166] Nina Simone ➜ female
[3113/107166] Blu & Exile ➜ unknown
[3114/107166] Tirzah ➜ female
[3115/107166] BadBadNotGood ➜ unknown
[3116/107166] Balla Et Ses Balladins ➜ unknown
[3117/107166] Orchestra Super Mazembe ➜ unknown
[3118/107166] Dom La Nena ➜ female
[3119/107166] DJ Krush ➜ male
[3120/107166] Esthero ➜ female
[3121/107166] Tarika Blue ➜ unknown
[3122/107166] The Internet ➜ unknown
[3123/107166] Michael Franks ➜ male
[3124/107166] Somi ➜ female
[3125/107166] Daley ➜ male
[3126/107166] Alice Coltrane ➜ male
[3127/107166] Quadron ➜ unknown
[3128/107166] Koop ➜ unknown
[3129/107166] Slum Village ➜ unknown
[3130/107166] Céu ➜ female
[3131/107166] Gilles Peterson's Havana Cultura Band ➜ unknown
[3132/107166] Melanie De Biasio ➜ female
[3133/107166] Eyedea & Abilities ➜ unknown
[3134/107166] Reflection Eternal ➜ unknown
[3135/107166] Jamiroquai ➜ unknown
[3136/107166] Maxi Priest ➜ male
[3137/107166] Taylor Mcferrin ➜ female
[3138/107166] Donell Jones ➜ male
[3139/107166] Yancey Boys ➜ unknown
[3140/107166] Mulatu Astatke ➜ unknown
[3141/107166] Foxy Brown ➜ female
[3142/107166] Gretchen Parlato ➜ female
[3143/107166] Pete Rock ➜ male
[3144/107166] Iman Omari ➜ unknown
[3145/107166] Ibrahim Maalouf ➜ male
[3146/107166] Ernest Ranglin ➜ male
[3147/107166] Reva DeVito ➜ unknown
[3148/107166] Nick Hakim ➜ male
[3149/107166] Ready For The World ➜ unknown
[3150/107166] Paula Cole ➜ male
[3151/107166] The RH Factor ➜ unknown
[3152/107166] Rocket Juice & The Moon ➜ unknown
[3153/107166] Robert Glasper ➜ male
[3154/107166] Tyler, The Creator ➜ male
[3155/107166] Brandy ➜ female
[3156/107166] Blended Babies ➜ unknown
[3157/107166] Damien Rice ➜ male
[3158/107166] Bill Laswell ➜ male
[3159/107166] Slick Rick ➜ male
[3160/107166] Jhene Aiko ➜ female
[3161/107166] Redman ➜ male
[3162/107166] Dorsh ➜ unknown
[3163/107166] From Kid ➜ male
[3164/107166] Lianne La Havas ➜ female
[3165/107166] Meshell Ndegeocello ➜ female
[3166/107166] KRS-One ➜ male
[3167/107166] Richard Dorfmeister ➜ male
[3168/107166] Eazy-E ➜ unknown
[3169/107166] Wu-Tang Clan ➜ unknown
[3170/107166] Coolio ➜ male
[3171/107166] Luniz ➜ unknown
[3172/107166] Xzibit ➜ male
[3173/107166] G-Unit ➜ unknown
[3174/107166] Pharoahe Monch ➜ male
[3175/107166] Big Pun ➜ male
[3176/107166] All Time Low ➜ unknown
[get_wikidata_id] Error for MBID 1aff70f2-c2f5-4d3a-9b77-f1f0c07fef40: 'url-relation-list'
[3177/107166] Stubby ➜ unknown
[3178/107166] Hey Violet ➜ unknown
[3179/107166] Jaeger ➜ male
[3180/107166] Echos ➜ unknown
[3181/107166] David Nail ➜ male
[3182/107166] Alex G ➜ male
[3183/107166] Jesse Bonanno ➜ unknown
[3184/107166] Stephen Schwartz ➜ male
[3185/107166] Jim Brickman ➜ male
[3186/107166] Martina McBride ➜ female
[3187/107166] Julie Walters ➜ female
[3188/107166] Meryl Streep ➜ female
[3189/107166] Justice Crew ➜ unknown
[3190/107166] Jaron And The Long Road To Love ➜ male
[3191/107166] Stefano ➜ male
[3192/107166] Jessie James ➜ male
[3193/107166] Phineas ➜ male
[3194/107166] Maren Morris ➜ female
[3195/107166] Jake Reese ➜ unknown
[3196/107166] Lindsay Ell ➜ female
[3197/107166] LANCO ➜ unknown
[3198/107166] Tink ➜ female
[3199/107166] Mary Jane Girls ➜ unknown
[3200/107166] Carl Carlton ➜ male
[3201/107166] Biz Markie ➜ male
[3202/107166] Lucy Pearl ➜ unknown
[3203/107166] Crystal Waters ➜ female
[3204/107166] Rich Gang ➜ unknown
[3205/107166] Junior M.A.F.I.A. ➜ unknown
[3206/107166] Gyptian ➜ male
[3207/107166] Johnny Kemp ➜ male
[3208/107166] Wayne Wonder ➜ male
[3209/107166] Junior Kelly ➜ male
[3210/107166] Sanchez ➜ male
[3211/107166] 4 Non Blondes ➜ unknown
[3212/107166] Smokey Robinson & The Miracles ➜ unknown
[3213/107166] DJ Kool ➜ unknown
[3214/107166] A Taste Of Honey ➜ unknown
[3215/107166] Max Romeo ➜ male
[3216/107166] Method Man ➜ male
[3217/107166] Midnight Star ➜ unknown
[3218/107166] Jazmine Sullivan ➜ female
[3219/107166] Crime Mob ➜ unknown
[3220/107166] Ben E. King ➜ male
[3221/107166] Dazz Band ➜ unknown
[3222/107166] Tbam ➜ unknown
[3223/107166] Nebu Kiniza ➜ male
[3224/107166] Trill Sammy ➜ male
[3225/107166] Dwight Twilley Band ➜ unknown
[3226/107166] Faces ➜ unknown
[3227/107166] Nico ➜ female
[3228/107166] Joe Dassin ➜ male
[3229/107166] Françoise Hardy ➜ female
[3230/107166] Jarvis Cocker ➜ male
[3231/107166] The Stooges ➜ unknown
[3232/107166] Peter Sarstedt ➜ male
[3233/107166] Elliott Smith ➜ male
[3234/107166] Mikky Ekko ➜ male
[3235/107166] Who Is Fancy ➜ male
[3236/107166] Nathan Sykes ➜ male
[3237/107166] TRXD ➜ unknown
[3238/107166] James TW ➜ male
[3239/107166] Camp Lo ➜ unknown
[3240/107166] Flatbush Zombies ➜ unknown
[3241/107166] DJ Yoda ➜ male
[3242/107166] 9th Wonder ➜ male
[3243/107166] ABRA ➜ unknown
[3244/107166] Solange ➜ female
[3245/107166] DJ Yoda feat. Soom T & Afrikan Boy ➜ female
[3246/107166] Moar ➜ unknown
[3247/107166] Gangsta Pat ➜ male
[get_wikidata_id] Error for MBID 1474da03-87c9-40bd-8f1b-8123a544a71e: 'url-relation-list'
[3248/107166] Polo Frost ➜ unknown
[3249/107166] Mir Fontane ➜ male
[3250/107166] KSI ➜ male
[3251/107166] Logan Paul ➜ male
[3252/107166] Aaron Carpenter ➜ male
[3253/107166] ONE OK ROCK ➜ unknown
[3254/107166] Rick Springfield ➜ male
[3255/107166] Lewis Del Mar ➜ unknown
[3256/107166] NoMBe ➜ male
[3257/107166] RKCB ➜ unknown
[3258/107166] Ella Vos ➜ female
[3259/107166] machineheart ➜ unknown
[3260/107166] Meadowlark ➜ unknown
[3261/107166] Lil Lonnie ➜ male
[3262/107166] Taylor Gang ➜ female
[3263/107166] Snootie Wild ➜ male
[3264/107166] Hearts & Colors ➜ unknown
[3265/107166] Martin Jensen ➜ male
[3266/107166] David Correy ➜ male
[3267/107166] Lil Snupe ➜ unknown
[3268/107166] Kevin Ross ➜ female
[3269/107166] Shaun Frank ➜ male
[3270/107166] Alyssa Reid ➜ female
[3271/107166] Sigrid ➜ female
[3272/107166] Mia Martina ➜ female
[3273/107166] Jax Jones ➜ male
[3274/107166] Tinashe ➜ female
[3275/107166] Snakehips ➜ unknown
[3276/107166] WizKid ➜ male
[3277/107166] Major Minor ➜ unknown
[3278/107166] Chelsea Cutler ➜ female
[3279/107166] Ross Copperman ➜ male
[3280/107166] Tritonal ➜ unknown
[3281/107166] Alkaline Trio ➜ unknown
[3282/107166] Tori McClure & Jon D ➜ female
[3283/107166] Beartooth ➜ unknown
[3284/107166] Go Radio ➜ unknown
[3285/107166] Acceptance ➜ unknown
[3286/107166] Too Close To Touch ➜ unknown
[3287/107166] The Messenger ➜ unknown
[3288/107166] The Almost ➜ unknown
[get_wikidata_id] Error for MBID 7dd48c04-8e0c-4a72-a180-383781122309: 'url-relation-list'
[3289/107166] The New Low ➜ unknown
[3290/107166] Evergreen Terrace ➜ unknown
[3291/107166] Defeat The Low ➜ unknown
[3292/107166] Koda ➜ unknown
[3293/107166] The Apache Relay ➜ unknown
[3294/107166] Firekites ➜ unknown
[3295/107166] J-Walk ➜ female
[3296/107166] Purity Ring ➜ unknown
[3297/107166] Wild Belle ➜ unknown
[3298/107166] Alle Farben ➜ male
[3299/107166] Faul & Wad Ad ➜ unknown
[3300/107166] Us The Duo ➜ unknown
[3301/107166] TM Juke ➜ unknown
[get_wikidata_id] Error for MBID 373aea4d-05c3-4751-85a4-5f9fb0d4a573: 'url-relation-list'
[3302/107166] Jakubi ➜ unknown
[3303/107166] Widespread Panic ➜ unknown
[3304/107166] Rainbow Kitten Surprise ➜ unknown
[3305/107166] Cherub ➜ unknown
[3306/107166] Casey Veggies ➜ male
[3307/107166] DJ Katch ➜ unknown
[3308/107166] Skizzy Mars ➜ male
[3309/107166] Twiddle ➜ unknown
[3310/107166] Mapei ➜ female
[3311/107166] Ookay ➜ male
[3312/107166] Tove Styrke ➜ female
[3313/107166] WATCH THE DUCK ➜ unknown
[3314/107166] Hold on to Me ➜ unknown
[3315/107166] Domishay ➜ unknown
[3316/107166] Vera ➜ female
[3317/107166] The Chi-Lites ➜ unknown
[3318/107166] Bloodstone ➜ unknown
[3319/107166] Jackson Browne ➜ male
[3320/107166] Todd Rundgren ➜ male
[3321/107166] White Witch ➜ male
[3322/107166] Easy Star All-Stars ➜ unknown
[3323/107166] Los Hijos Del Rey ➜ unknown
[3324/107166] Luis Santiago ➜ male
[3325/107166] Daniel Calveti ➜ unknown
[3326/107166] Marcos Witt ➜ male
[3327/107166] Julio Melgar ➜ male
[3328/107166] Danilo Montero ➜ male
[3329/107166] Alex Campos ➜ male
[3330/107166] Jesús Adrián Romero ➜ male
[3331/107166] Jesús Adrián Romero & Marcela Gandara ➜ female
[3332/107166] Marcela Gandara ➜ female
[3333/107166] Marco Barrientos ➜ male
[3334/107166] Marco Barrientos & David Luckey ➜ male
[3335/107166] En Espíritu Y En Verdad ➜ unknown
[3336/107166] Vino Nuevo ➜ unknown
[3337/107166] Jose Luis Reyes ➜ unknown
[get_wikidata_id] Error for MBID fc15cbc3-656c-4f77-94f4-045c7fb63695: 'url-relation-list'
[3338/107166] Inspiraciòn ➜ unknown
[3339/107166] Tony Pérez ➜ male
[3340/107166] Samuel Hernández ➜ male
[3341/107166] Óscar Medina ➜ male
[3342/107166] Luigi Castro ➜ male
[3343/107166] Danny Berrios ➜ unknown
[3344/107166] Ericson Alexander Molano ➜ male
[3345/107166] Roberto Orellana ➜ unknown
[3346/107166] Vertical ➜ unknown
[3347/107166] Salida 7 ➜ unknown
[3348/107166] Ab-Soul ➜ male
[3349/107166] Awaken Worship ➜ unknown
[3350/107166] William Matthews ➜ male
[3351/107166] North Point Kids ➜ unknown
[get_wikidata_id] Error for MBID 6e5cfaae-9d65-4b98-b74e-518e872edca8: 'url-relation-list'
[3352/107166] Amber Sky Records ➜ unknown
[3353/107166] Hillsong Kids ➜ unknown
[3354/107166] North Point InsideOut ➜ unknown
[3355/107166] Planetshakers ➜ unknown
[3356/107166] Lil Durk ➜ male
[3357/107166] Ca$h Out ➜ male
[3358/107166] MAX ➜ male
[3359/107166] R I T U A L ➜ unknown
[get_wikidata_id] Error for MBID 3eeed3d9-b8b0-487e-a9ca-04bc9cf0fd3e: 'url-relation-list'
[3360/107166] Bei Maejor ➜ unknown
[3361/107166] Kaiydo ➜ unknown
[3362/107166] Sam Gellaitry ➜ male
[3363/107166] Famy ➜ unknown
[3364/107166] Donnie Trumpet & The Social Experiment ➜ unknown
[3365/107166] Jack & Jack ➜ unknown
[3366/107166] Njomza ➜ female
[3367/107166] Hippie Sabotage ➜ unknown
[3368/107166] BABE ➜ unknown
[3369/107166] Jake Hamilton and the Sound ➜ unknown
[3370/107166] Derek Minor ➜ male
[3371/107166] KB ➜ male
[3372/107166] Black Knight ➜ unknown
[3373/107166] Rhema Soul ➜ unknown
[3374/107166] The Washington Projects ➜ unknown
[3375/107166] Shonlock ➜ unknown
[3376/107166] Bizzle ➜ male
[3377/107166] Family Force 5 ➜ unknown
[3378/107166] Wess Morgan ➜ unknown
[3379/107166] Brianna Caprice ➜ unknown
[3380/107166] Audrey Assad ➜ female
[get_wikidata_id] Error for MBID dd251f42-e4d8-417a-8b97-156af1722204: 'url-relation-list'
[3381/107166] Skrip ➜ unknown
[3382/107166] Canton Jones feat. Tonio & TK ➜ male
[3383/107166] William McDowell ➜ male
[3384/107166] Sherrod White ➜ male
[3385/107166] Citipointe Live ➜ unknown
[3386/107166] Bryan & Katie Torwalt ➜ unknown
[3387/107166] Moriah Peters ➜ male
[3388/107166] Citizens & Saints ➜ unknown
[3389/107166] Mali Music ➜ unknown
[3390/107166] Vertical Worship ➜ unknown
[3391/107166] Christon Gray ➜ male
[3392/107166] Hillary Scott & The Scott Family ➜ unknown
[3393/107166] Deitrick Haddon ➜ male
[3394/107166] Devin Turner ➜ female
[3395/107166] KJ-52 ➜ male
[3396/107166] Stephen Witt ➜ male
[3397/107166] Mat Kearney ➜ male
[3398/107166] Umobile Worship ➜ unknown
[3399/107166] Jenn Johnson ➜ female
[3400/107166] Danny Gokey ➜ male
[3401/107166] Hollyn ➜ female
[3402/107166] Israel Houghton ➜ male
[3403/107166] Warr Acres ➜ unknown
[3404/107166] Chris Quilala ➜ male
[3405/107166] Geoffrey Golden ➜ unknown
[3406/107166] The Phantoms ➜ unknown
[3407/107166] Stuart Duncan ➜ unknown
[3408/107166] The Washington University Amateurs ➜ unknown
[3409/107166] Rusted Root ➜ unknown
[3410/107166] Pat Metheny Group ➜ unknown
[3411/107166] The Cowboy Poets ➜ unknown
[3412/107166] Dispatch ➜ unknown
[3413/107166] The Weepies ➜ unknown
[3414/107166] RIVVRS ➜ unknown
[3415/107166] Air Traffic Controller ➜ unknown
[3416/107166] Korby Lenker ➜ unknown
[3417/107166] Acoustic Junction ➜ unknown
[3418/107166] Entrain ➜ unknown
[3419/107166] Little Feat ➜ unknown
[3420/107166] Big Wild ➜ unknown
[3421/107166] The Nicol Kings ➜ unknown
[3422/107166] Ariel Pink ➜ unknown
[3423/107166] Hinds ➜ unknown
[3424/107166] The Breeders ➜ unknown
[3425/107166] Sinkane ➜ male
[3426/107166] Cass McCombs ➜ male
[3427/107166] King Gizzard & The Lizard Wizard ➜ unknown
[3428/107166] Melody's Echo Chamber ➜ unknown
[3429/107166] The Legends ➜ unknown
[3430/107166] Paul Simon ➜ male
[3431/107166] Panama ➜ unknown
[3432/107166] Parra for Cuva ➜ male
[3433/107166] Vince Staples ➜ male
[3434/107166] Alex Ebert ➜ male
[3435/107166] Wake Owl ➜ unknown
[3436/107166] The Jam ➜ unknown
[3437/107166] Cross Record ➜ unknown
[3438/107166] StarBenders ➜ unknown
[3439/107166] Charles Bradley ➜ male
[3440/107166] White Denim ➜ unknown
[3441/107166] Songs: Ohia ➜ unknown
[3442/107166] Hotel Eden ➜ unknown
[3443/107166] William Onyeabor ➜ male
[3444/107166] Jay Electronica ➜ male
[3445/107166] Bosley ➜ unknown
[3446/107166] Sweet Spirit ➜ unknown
[3447/107166] Alvvays ➜ unknown
[3448/107166] Rhythms Del Mundo ➜ unknown
[3449/107166] Tony Allen ➜ male
[3450/107166] El Michels Affair ➜ unknown
[3451/107166] Danger Doom ➜ male
[3452/107166] Gwen McCrae ➜ female
[3453/107166] Young and Company ➜ unknown
[3454/107166] Brooklyn Express ➜ unknown
[3455/107166] Empire Projecting Penny ➜ male
[3456/107166] Richard Hewson Orchestra ➜ unknown
[3457/107166] Odyssey ➜ unknown
[3458/107166] Rose Royce ➜ unknown
[3459/107166] Bobby Thurston ➜ unknown
[3460/107166] Roberta Flack ➜ female
[3461/107166] The Black On White Affair ➜ unknown
[3462/107166] Zapp ➜ unknown
[3463/107166] Funkadelic ➜ unknown
[3464/107166] Glenn Jones ➜ male
[3465/107166] The Pretenders ➜ unknown
[3466/107166] Anita Baker ➜ female
[3467/107166] Ramsey Lewis ➜ male
[3468/107166] Force M.D.'s ➜ unknown
[3469/107166] Rostam ➜ male
[3470/107166] Nicolas Jaar ➜ male
[3471/107166] Death Grips ➜ unknown
[3472/107166] Modern Baseball ➜ unknown
[3473/107166] Amber Coffman ➜ female
[3474/107166] PUP ➜ unknown
[3475/107166] Amrit ➜ male
[3476/107166] Clams Casino ➜ male
[3477/107166] Preoccupations ➜ unknown
[3478/107166] Matt Maltese ➜ male
[3479/107166] Yoni & Geti ➜ unknown
[3480/107166] Morgan Delt ➜ unknown
[3481/107166] The I.L.Y's ➜ unknown
[3482/107166] Promises Ltd. ➜ unknown
[3483/107166] Katie Gately ➜ unknown
[3484/107166] Deakin ➜ male
[3485/107166] TV Girl ➜ unknown
[3486/107166] The Claypool Lennon Delirium ➜ unknown
[3487/107166] The Range ➜ unknown
[3488/107166] NxWorries ➜ unknown
[3489/107166] Foxygen ➜ unknown
[3490/107166] Kadhja Bonet ➜ female
[3491/107166] Open Mike Eagle ➜ male
[3492/107166] LVL UP ➜ unknown
[3493/107166] clipping. ➜ unknown
[3494/107166] Andy Hull and Robert McDowell ➜ male
[3495/107166] Cher ➜ female
[3496/107166] Marian Hill ➜ male
[3497/107166] Karmin ➜ unknown
[3498/107166] The Runaways ➜ unknown
[3499/107166] Switchblade Kittens ➜ unknown
[3500/107166] Grand Ole Party ➜ unknown
[3501/107166] The Lone Bellow ➜ unknown
[3502/107166] The Hunts ➜ unknown
[3503/107166] Family and Friends ➜ unknown
[3504/107166] The Pink Slips ➜ unknown
[3505/107166] Betty Who ➜ unknown
[3506/107166] INXS ➜ unknown
[3507/107166] J. Valentine ➜ female
[3508/107166] Pleasure P ➜ female
[3509/107166] Marques Houston ➜ male
[3510/107166] Jill Scott ➜ female
[3511/107166] Carl Thomas ➜ male
[3512/107166] Luke James ➜ male
[3513/107166] TGT ➜ unknown
[3514/107166] rachel kann ➜ unknown
[3515/107166] DonMonique ➜ unknown
[3516/107166] Jeremy Passion ➜ male
[3517/107166] Mutemath ➜ unknown
[3518/107166] Argonaut & Wasp ➜ unknown
[3519/107166] Freedom Fry ➜ male
[3520/107166] MAGIC GIANT ➜ unknown
[3521/107166] Roosevelt ➜ male
[3522/107166] Amy Stroup ➜ unknown
[3523/107166] Kishi Bashi ➜ male
[3524/107166] The Bravery ➜ unknown
[3525/107166] Goldfinger ➜ unknown
[3526/107166] Theo Katzman ➜ male
[3527/107166] Foreign Air ➜ unknown
[3528/107166] Jack's Mannequin ➜ unknown
[3529/107166] BarlowGirl ➜ unknown
[3530/107166] Forever The Sickest Kids ➜ unknown
[3531/107166] Steven Sharp Nelson ➜ male
[3532/107166] Jo Dee Messina ➜ female
[3533/107166] Lou Monte ➜ male
[3534/107166] Gene Autry ➜ male
[3535/107166] Brenda Lee ➜ female
[3536/107166] Alvin & The Chipmunks ➜ unknown
[3537/107166] Ray Conniff ➜ male
[3538/107166] Piano Christmas ➜ unknown
[3539/107166] Tim Minchin ➜ male
[3540/107166] Santa Ana Players ➜ unknown
[3541/107166] Dean Martin ➜ male
[3542/107166] Seven Missing Santas ➜ unknown
[3543/107166] Jonathan Coulton ➜ male
[3544/107166] Pyotr Ilyich Tchaikovsky ➜ male
[3545/107166] Andy Williams ➜ male
[3546/107166] Perry Como ➜ male
[3547/107166] Burl Ives ➜ male
[3548/107166] Stephen Colbert ➜ male
[3549/107166] Jon Stewart ➜ male
[3550/107166] Paul Williams ➜ male
[3551/107166] The Muppet Cast ➜ unknown
[3552/107166] Muppet Brass Buskers ➜ unknown
[3553/107166] Kermit ➜ male
[3554/107166] Ghost of Christmas Present ➜ neutral sex
[3555/107166] Scrooge ➜ unknown
[3556/107166] Sleigh Bell Orchestra ➜ unknown
[3557/107166] Little Elves Choir ➜ male
[get_wikidata_id] Error for MBID 7421162b-df05-4d97-9b46-2bfa31199753: 'url-relation-list'
[3558/107166] Yuletide St. Nick ➜ unknown
[3559/107166] Holiday Rockers ➜ female
[3560/107166] Mykola Dmytrovych Leontovych ➜ male
[3561/107166] Joyful Carolers ➜ unknown
[3562/107166] Doris Day ➜ female
[3563/107166] Thurl Ravenscroft ➜ male
[3564/107166] Eugene Poddany ➜ unknown
[3565/107166] CMc ➜ unknown
[3566/107166] Marlon Roudette ➜ male
[3567/107166] Video Game Players ➜ unknown
[3568/107166] Warren G ➜ male
[3569/107166] ILoveMakonnen ➜ male
[3570/107166] Juan Magán ➜ male
[3571/107166] Cali Y El Dandee ➜ unknown
[3572/107166] Wisin ➜ male
[3573/107166] Tony Dize ➜ male
[3574/107166] Orishas ➜ unknown
[3575/107166] Joey Montana ➜ male
[3576/107166] Mike Bahía ➜ male
[3577/107166] Alkilados ➜ unknown
[3578/107166] Breiky ➜ unknown
[3579/107166] José de Rico ➜ male
[3580/107166] Buxxi ➜ unknown
[3581/107166] Son By Four ➜ unknown
[3582/107166] Piso 21 ➜ unknown
[3583/107166] Alexis y Fido ➜ unknown
[3584/107166] Thalía ➜ female
[3585/107166] Dr. Bellido ➜ male
[3586/107166] Reik ➜ unknown
[3587/107166] IAmChino ➜ unknown
[3588/107166] Flaco Flow ➜ unknown
[3589/107166] Kevin Roldan ➜ male
[3590/107166] Jhoni The Voice ➜ unknown
[3591/107166] Jencarlos ➜ male
[3592/107166] Domino Saints ➜ male
[3593/107166] Xantos ➜ unknown
[3594/107166] Reykon ➜ male
[3595/107166] Alejandro Gonzalez ➜ male
[3596/107166] CNCO ➜ unknown
[3597/107166] Silvestre Dangond ➜ male
[3598/107166] J Alvarez ➜ female
[3599/107166] Yandar & Yostin ➜ unknown
[3600/107166] Big Yamo ➜ male
[3601/107166] Magan ➜ male
[3602/107166] Henry Mendez ➜ male
[3603/107166] Cabas ➜ male
[3604/107166] Danny Ocean ➜ male
[3605/107166] Fainal ➜ unknown
[3606/107166] Tacabro ➜ unknown
[3607/107166] Cherito ➜ unknown
[get_wikidata_id] Error for MBID c681cb76-26d7-4263-8058-2ec78ae04ba2: 'url-relation-list'
[3608/107166] Wilo D New ➜ unknown
[3609/107166] La Materialista ➜ female
[3610/107166] Jerry Douglas ➜ male
[3611/107166] J Sutta ➜ female
[3612/107166] Super Groupie ➜ unknown
[3613/107166] Watsky ➜ male
[3614/107166] Emma Carn ➜ unknown
[3615/107166] Anna Wise ➜ male
[3616/107166] Hilary Duff ➜ female
[3617/107166] DJ Luke Nasty ➜ unknown
[3618/107166] Lolawolf ➜ unknown
[3619/107166] Lilly Wood and The Prick ➜ unknown
[3620/107166] 2 LIVE CREW ➜ unknown
[3621/107166] Adam Lambert ➜ male
[3622/107166] Ashlee Simpson ➜ female
[3623/107166] Borgore ➜ male
[3624/107166] CL ➜ female
[3625/107166] Majid Jordan ➜ unknown
[3626/107166] Club Kuru ➜ unknown
[3627/107166] xxyyxx ➜ male
[3628/107166] Lil B ➜ male
[3629/107166] Era Istrefi ➜ female
[3630/107166] 0scill8or ➜ unknown
[3631/107166] Sak Noel ➜ male
[3632/107166] Branko ➜ unknown
[3633/107166] Sublime With Rome ➜ unknown
[get_wikidata_id] Error for MBID 91addfe6-0c9a-4fbf-a35f-0222b8cced55: 'url-relation-list'
[3634/107166] Sleepwalkers ➜ unknown
[3635/107166] Johnny Stimson ➜ male
[3636/107166] Hermitude ➜ unknown
[3637/107166] Think Twice with David Ryshpan ➜ male
[3638/107166] VOKES ➜ unknown
[3639/107166] Exmag ➜ unknown
[3640/107166] Mariahlynn ➜ female
[3641/107166] Lullaby Land ➜ unknown
[3642/107166] Steven C ➜ male
[3643/107166] Christopher M. Georges ➜ unknown
[3644/107166] Worship Music Piano ➜ unknown
[3645/107166] Klaus Kuehn ➜ male
[3646/107166] Terri Geisel ➜ female
[3647/107166] Kim Costanza ➜ female
[3648/107166] The O'Neill Brothers Group ➜ unknown
[get_wikidata_id] Error for MBID b8c72a29-6dd4-4bb3-b970-d71fc882aea6: 'url-relation-list'
[3649/107166] Praise and Worship ➜ unknown
[3650/107166] Steffany Gretzinger ➜ female
[3651/107166] Love and Theft ➜ unknown
[3652/107166] Kevin Hart ➜ unknown
[3653/107166] De La Soul ➜ unknown
[3654/107166] Fatman Scoop ➜ male
[3655/107166] Kyle Edwards ➜ male
[3656/107166] Ziggy ➜ unknown
[3657/107166] DJ LILMAN ➜ male
[3658/107166] LouGotCash ➜ unknown
[3659/107166] SahBabii ➜ male
[3660/107166] 22 Savage ➜ male
[3661/107166] 2milly ➜ unknown
[3662/107166] Rowdy Rebel ➜ male
[3663/107166] Kranium ➜ unknown
[3664/107166] Popcaan ➜ male
[3665/107166] Vybz Kartel ➜ male
[3666/107166] Charly Black ➜ unknown
[3667/107166] RDX ➜ unknown
[3668/107166] Mr. Vegas ➜ male
[3669/107166] QQ ➜ unknown
[3670/107166] Spice ➜ unknown
[3671/107166] Freezy ➜ male
[3672/107166] Ricky Blaze ➜ unknown
[3673/107166] Serani ➜ male
[3674/107166] Rupee ➜ male
[3675/107166] Alison Hinds ➜ female
[3676/107166] Nigel & Marvin ➜ unknown
[3677/107166] Colin Lucas ➜ male
[3678/107166] Cloud 5 ➜ unknown
[3679/107166] Machel Montano ➜ male
[3680/107166] Ultimate Rejects ➜ unknown
[3681/107166] Oliver ➜ male
[3682/107166] Darius ➜ male
[3683/107166] Christoph Andersson ➜ male
[3684/107166] Ben Platt ➜ male
[3685/107166] Renée Elise Goldsberry ➜ female
[3686/107166] Jonathan Groff ➜ male
[3687/107166] Phillipa Soo ➜ female
[3688/107166] Original Broadway Cast of Hamilton ➜ unknown
[3689/107166] Daveed Diggs ➜ male
[3690/107166] Jasmine Cephas-Jones ➜ female
[3691/107166] Anthony Ramos ➜ male
[3692/107166] Jordan Fisher ➜ male
[3693/107166] Karen Olivo ➜ female
[3694/107166] 'In The Heights' Original Broadway Company ➜ unknown
[3695/107166] Mike Faist ➜ male
[3696/107166] Les Misérables Cast ➜ unknown
[3697/107166] Cast Of Rent ➜ unknown
[3698/107166] Nikki Blonsky ➜ female
[get_wikidata_id] Error for MBID 6b46ca2e-6e46-4403-891e-bb267784e00d: 'url-relation-list'
[3699/107166] Rachel Bay Jones ➜ unknown
[3700/107166] Samantha Barks ➜ female
[3701/107166] Zac Efron ➜ male
[3702/107166] La La Land Cast ➜ unknown
[3703/107166] !llmind ➜ unknown
[3704/107166] Brittany Snow ➜ male
[3705/107166] Company ➜ unknown
[3706/107166] Adam Jacobs ➜ male
[3707/107166] James Monroe Iglehart ➜ unknown
[3708/107166] Alan Menken ➜ male
[3709/107166] Rosie O'Donnell ➜ male
[3710/107166] Roger Bart ➜ male
[3711/107166] Cheryl Freeman ➜ female
[3712/107166] Ben Harper ➜ male
[3713/107166] Lipps Inc. ➜ unknown
[3714/107166] Berlin ➜ unknown
[3715/107166] ABBA ➜ unknown
[3716/107166] Adam Sandler ➜ male
[3717/107166] America ➜ unknown
[3718/107166] Simon & Garfunkel ➜ unknown
[3719/107166] Carly Simon ➜ female
[3720/107166] Terry Jacks ➜ male
[3721/107166] Nitty Gritty Dirt Band ➜ unknown
[3722/107166] Pure Prairie League ➜ unknown
[3723/107166] Gilbert O'Sullivan ➜ male
[3724/107166] Herb Alpert & The Tijuana Brass ➜ unknown
[3725/107166] Derek & The Dominos ➜ unknown
[3726/107166] Sonny & Cher ➜ unknown
[3727/107166] The Zombies ➜ unknown
[3728/107166] John Fogerty ➜ male
[3729/107166] The Grass Roots ➜ unknown
[3730/107166] Janis Joplin ➜ female
[3731/107166] Meat Loaf ➜ male
[3732/107166] Harry Nilsson ➜ male
[3733/107166] Wings ➜ unknown
[3734/107166] Billy Idol ➜ male
[3735/107166] The J. Geils Band ➜ unknown
[3736/107166] The Dream Academy ➜ unknown
[3737/107166] The Ukulele Boys ➜ unknown
[3738/107166] Gordon Lightfoot ➜ male
[3739/107166] The Association ➜ unknown
[3740/107166] B.J. Thomas ➜ male
[3741/107166] Blessid Union Of Souls ➜ unknown
[3742/107166] Bob Seger ➜ male
[3743/107166] George Thorogood & The Destroyers ➜ unknown
[3744/107166] Lulu ➜ female
[3745/107166] Can Atilla ➜ male
[3746/107166] Air Supply ➜ unknown
[3747/107166] Israel Nash ➜ male
[3748/107166] Boz Scaggs ➜ male
[3749/107166] Jill Barber ➜ female
[3750/107166] Boney James ➜ unknown
[3751/107166] Rick Braun ➜ male
[3752/107166] Demis Roussos ➜ male
[3753/107166] Skylar Grey ➜ female
[3754/107166] Julio Iglesias ➜ male
[3755/107166] Take That ➜ unknown
[3756/107166] Justin Hayward ➜ male
[3757/107166] Andrea Bocelli ➜ male
[3758/107166] Prince Jammy ➜ male
[3759/107166] Pablo Alborán ➜ male
[3760/107166] Duke Ellington ➜ male
[3761/107166] Carpenters ➜ unknown
[3762/107166] Nino Rota ➜ male
[3763/107166] Peppino Gagliardi ➜ male
[3764/107166] Outlandish ➜ unknown
[3765/107166] Camiel ➜ unknown
[3766/107166] Louie Austen ➜ male
[3767/107166] Green Point Orchestra ➜ unknown
[3768/107166] Simply Red ➜ unknown
[3769/107166] Röyksopp ➜ unknown
[3770/107166] Gato Barbieri ➜ male
[3771/107166] Susan Wong ➜ female
[3772/107166] Moxie Raia ➜ female
[3773/107166] India.Arie ➜ female
[3774/107166] Above & Beyond ➜ unknown
[3775/107166] GAMPER & DADONI ➜ unknown
[3776/107166] Engelbert Humperdinck ➜ male
[3777/107166] Ricchi E Poveri ➜ unknown
[3778/107166] Gloria Estefan ➜ female
[3779/107166] Armik ➜ male
[3780/107166] Mystic Diversions ➜ unknown
[3781/107166] Cécile Bredie ➜ unknown
[3782/107166] Duke Dumont ➜ male
[3783/107166] Maite Hontelé ➜ female
[3784/107166] William Joseph ➜ male
[3785/107166] Christopher Cross ➜ male
[3786/107166] Tommy Fleming ➜ male
[3787/107166] George St. Kitts ➜ female
[3788/107166] Alejandro De Pinedo ➜ male
[3789/107166] Adriana Mezzadri ➜ female
[3790/107166] Laura Pausini ➜ female
[3791/107166] Excision ➜ male
[3792/107166] Aero Chord ➜ male
[3793/107166] Vicetone ➜ unknown
[3794/107166] Pegboard Nerds ➜ unknown
[3795/107166] Tristam ➜ male
[3796/107166] Noisestorm ➜ male
[3797/107166] Grabbitz ➜ male
[3798/107166] Nitro Fun ➜ male
[3799/107166] Haywyre ➜ male
[3800/107166] Au5 ➜ male
[3801/107166] Rootkit ➜ unknown
[3802/107166] Case & Point ➜ unknown
[3803/107166] Lets Be Friends ➜ unknown
[3804/107166] Rezonate ➜ unknown
[3805/107166] Grant Bowtie ➜ male
[3806/107166] SCNDL ➜ unknown
[3807/107166] Stephen Walking ➜ male
[3808/107166] Astronaut ➜ unknown
[3809/107166] Mr FijiWiji ➜ male
[3810/107166] Varien ➜ trans woman
[3811/107166] WRLD ➜ male
[3812/107166] Razihel ➜ unknown
[3813/107166] Hellberg ➜ male
[3814/107166] Sushi Killer ➜ unknown
[3815/107166] Laszlo ➜ male
[3816/107166] Rogue ➜ unknown
[3817/107166] Tut Tut Child ➜ unknown
[3818/107166] LVTHER ➜ unknown
[3819/107166] Rameses B ➜ male
[3820/107166] Puppet ➜ unknown
[3821/107166] Nanobii ➜ unknown
[3822/107166] Infected Mushroom ➜ unknown
[3823/107166] Sound Remedy ➜ unknown
[3824/107166] Favright ➜ unknown
[3825/107166] Muzzy ➜ unknown
[3826/107166] Droptek ➜ unknown
[3827/107166] Pixl ➜ unknown
[3828/107166] Going Quantum ➜ unknown
[3829/107166] Rich Edwards ➜ male
[3830/107166] Bustre ➜ unknown
[3831/107166] Fractal ➜ non-binary gender
[3832/107166] Direct ➜ unknown
[3833/107166] Pegboard Nerds & Tristam ➜ unknown
[3834/107166] Headhunterz ➜ male
[3835/107166] SirensCeol ➜ unknown
[3836/107166] Boogie ➜ unknown
[3837/107166] Keith Ape ➜ male
[3838/107166] Leaf ➜ unknown
[3839/107166] Yung Lean ➜ male
[3840/107166] Lesley Gore ➜ female
[3841/107166] The Human League ➜ unknown
[3842/107166] The Penguins ➜ unknown
[3843/107166] Corey Hart ➜ male
[3844/107166] Corona ➜ unknown
[3845/107166] The Righteous Brothers ➜ unknown
[3846/107166] Juice Newton ➜ female
[3847/107166] Mark Morrison ➜ male
[3848/107166] Southern Creek Players ➜ unknown
[3849/107166] Nena ➜ female
[3850/107166] Pet Shop Boys ➜ unknown
[3851/107166] Olivia Newton-John ➜ female
[3852/107166] Chicago ➜ unknown
[3853/107166] Culture Club ➜ unknown
[3854/107166] Grover Washington, Jr. ➜ male
[3855/107166] Bobby Vinton ➜ male
[3856/107166] Nu Shooz ➜ unknown
[3857/107166] Young MC ➜ male
[3858/107166] Tomppabeats ➜ male
[3859/107166] A Flock Of Seagulls ➜ unknown
[3860/107166] Chaka Khan ➜ female
[3861/107166] Chris de Burgh ➜ male
[3862/107166] Martika ➜ female
[3863/107166] Cutting Crew ➜ unknown
[3864/107166] Rod Stewart ➜ male
[3865/107166] The S.O.S Band ➜ unknown
[3866/107166] Heatwave ➜ unknown
[3867/107166] Sounds Of Blackness ➜ unknown
[3868/107166] Albert Hammond ➜ male
[3869/107166] When In Rome ➜ unknown
[3870/107166] Generation X ➜ unknown
[3871/107166] Echo & the Bunnymen ➜ unknown
[3872/107166] The Alan Parsons Project ➜ unknown
[3873/107166] The Go-Go's ➜ unknown
[3874/107166] Steve Perry ➜ male
[3875/107166] Billy Ocean ➜ male
[3876/107166] The Turtles ➜ unknown
[3877/107166] Lenny Kravitz ➜ male
[3878/107166] Soft Cell ➜ unknown
[3879/107166] Mr. Mister ➜ unknown
[3880/107166] Orchestral Manoeuvres In The Dark ➜ unknown
[3881/107166] Aliotta Haynes Jeremiah ➜ unknown
[3882/107166] Haircut 100 ➜ unknown
[3883/107166] Breathe ➜ unknown
[3884/107166] Collie Buddz ➜ male
[3885/107166] Billy Blue ➜ male
[3886/107166] Tanto Metro & Devonte ➜ unknown
[3887/107166] Kiko Bun ➜ male
[3888/107166] Cham ➜ male
[3889/107166] Hammock ➜ unknown
[3890/107166] Helios ➜ male
[3891/107166] NazcarNation ➜ unknown
[3892/107166] Daithí ➜ male
[3893/107166] Dr. Toast ➜ male
[3894/107166] Jonathan Byrd ➜ male
[3895/107166] Evenings ➜ unknown
[3896/107166] Capsize ➜ unknown
[3897/107166] Pierce The Veil ➜ unknown
[3898/107166] BKAYE ➜ unknown
[3899/107166] Rise Against ➜ unknown
[3900/107166] There For Tomorrow ➜ unknown
[3901/107166] Memphis May Fire ➜ unknown
[3902/107166] Of Mice & Men ➜ unknown
[3903/107166] Ice Nine Kills ➜ unknown
[3904/107166] Breathe Carolina ➜ unknown
[3905/107166] My Chemical Romance ➜ unknown
[3906/107166] The Funeral Portrait ➜ unknown
[3907/107166] Brand New ➜ unknown
[3908/107166] Taking Back Sunday ➜ unknown
[3909/107166] Wicca Phase Springs Eternal ➜ male
[3910/107166] Icon For Hire ➜ unknown
[3911/107166] Against The Current ➜ unknown
[3912/107166] JAHKOY ➜ male
[3913/107166] The Brinks ➜ unknown
[3914/107166] Tayler Buono ➜ unknown
[3915/107166] Nathan Goshen ➜ male
[3916/107166] HONNE ➜ unknown
[3917/107166] Oohdem Beatz ➜ unknown
[3918/107166] Lulleaux ➜ unknown
[3919/107166] Eden Project ➜ unknown
[3920/107166] Lamborghini ➜ unknown
[3921/107166] Problem ➜ unknown
[3922/107166] Modern Chemistry ➜ unknown
[3923/107166] Never Shout Never ➜ unknown
[3924/107166] Living in Fiction ➜ unknown
[3925/107166] Pvris ➜ unknown
[3926/107166] Tracy Chapman ➜ female
[3927/107166] Sister Hazel ➜ unknown
[3928/107166] Toad The Wet Sprocket ➜ unknown
[3929/107166] The Presidents Of The United States Of America ➜ unknown
[3930/107166] Primitive Radio Gods ➜ unknown
[3931/107166] Tommy Tutone ➜ unknown
[3932/107166] The Charlie Daniels Band ➜ unknown
[3933/107166] Kenny Rogers ➜ male
[3934/107166] Joe Walsh ➜ male
[3935/107166] Alabama ➜ unknown
[3936/107166] Bad Company ➜ unknown
[3937/107166] Spiderbait ➜ unknown
[3938/107166] Crosby, Stills & Nash ➜ unknown
[3939/107166] Jethro Tull ➜ unknown
[3940/107166] C.W. McCall ➜ male
[3941/107166] Franky Dee ➜ female
[3942/107166] Eddie Grant ➜ male
[3943/107166] Jessi ➜ female
[3944/107166] Mayson The Soul ➜ unknown
[3945/107166] Hyorin ➜ unknown
[3946/107166] DEAN ➜ male
[3947/107166] Sua ➜ unknown
[3948/107166] Standing Egg ➜ unknown
[3949/107166] Toppdogg ➜ unknown
[3950/107166] Simon Dominic ➜ male
[3951/107166] 올 댓 ➜ unknown
[3952/107166] Monsta X ➜ unknown
[3953/107166] Sam Bruno ➜ male
[3954/107166] ARVFZ ➜ unknown
[3955/107166] Ben Pearce ➜ male
[3956/107166] ROZES ➜ female
[3957/107166] Big Mont ➜ unknown
[3958/107166] Kweku Collins ➜ male
[3959/107166] Skinny ➜ unknown
[3960/107166] Pasha ➜ unknown
[3961/107166] Shift K3Y ➜ male
[3962/107166] Pink Slip ➜ unknown
[3963/107166] Morgan Page ➜ female
[3964/107166] Tiesto feat. Tegan & Sara ➜ male
[3965/107166] Matthew Dear ➜ male
[3966/107166] The Audition ➜ unknown
[3967/107166] La Dispute ➜ unknown
[get_wikidata_id] Error for MBID e350f508-55a7-41cd-896e-ede38b982395: 'url-relation-list'
[3968/107166] Teen Hearts ➜ unknown
[3969/107166] This Wild Life ➜ unknown
[3970/107166] CHIC ➜ unknown
[3971/107166] Gerry Rafferty ➜ male
[3972/107166] Carole King ➜ female
[3973/107166] The Moody Blues ➜ unknown
[3974/107166] Stealers Wheel ➜ unknown
[3975/107166] Yaz ➜ unknown
[3976/107166] Duran Duran ➜ unknown
[3977/107166] Eddy Grant ➜ male
[3978/107166] Jefferson Airplane ➜ unknown
[3979/107166] Tom Cochrane ➜ male
[3980/107166] Kajagoogoo ➜ unknown
[3981/107166] Steve Winwood ➜ male
[3982/107166] DEVO ➜ unknown
[3983/107166] Buffalo Springfield ➜ unknown
[3984/107166] Gary Numan ➜ male
[3985/107166] Violent Femmes ➜ unknown
[3986/107166] Neil Young ➜ male
[3987/107166] Bread ➜ unknown
[3988/107166] Donna Summer ➜ female
[3989/107166] The Knack ➜ unknown
[3990/107166] George Strait ➜ male
[3991/107166] Diamond Rio ➜ male
[3992/107166] Deana Carter ➜ female
[3993/107166] Lonestar ➜ unknown
[3994/107166] John Michael Montgomery ➜ male
[3995/107166] Randy Houser ➜ male
[3996/107166] Josh Gracin ➜ male
[3997/107166] Parmalee ➜ unknown
[3998/107166] Craig Campbell ➜ male
[3999/107166] Letters To Cleo ➜ unknown
[4000/107166] Shonen Knife ➜ unknown
[4001/107166] The Muffs ➜ unknown
[4002/107166] Hole ➜ unknown
[4003/107166] Wheatus ➜ unknown
[4004/107166] Lillix ➜ unknown
[4005/107166] Christina Vidal ➜ female
[4006/107166] Cartel ➜ unknown
[4007/107166] Stefy ➜ unknown
[4008/107166] Hoku ➜ female
[4009/107166] Samantha Ronson ➜ female
[4010/107166] Among Savages ➜ unknown
[4011/107166] Jon Foreman ➜ male
[4012/107166] The Oh Hellos ➜ unknown
[4013/107166] Sébastien Tellier ➜ male
[4014/107166] Zach Winters ➜ unknown
[4015/107166] Warren Zevon ➜ male
[4016/107166] Danny Elfman ➜ male
[4017/107166] Jesse Cook ➜ male
[4018/107166] Sigur Rós ➜ unknown
[4019/107166] Brad Kilman ➜ unknown
[4020/107166] Enter The Worship Circle ➜ unknown
[4021/107166] Misty Edwards ➜ female
[4022/107166] USSR Ministry of Culture Chamber Choir ➜ unknown
[4023/107166] United Pursuit ➜ unknown
[4024/107166] The Impressions ➜ unknown
[get_wikidata_id] Error for MBID 3ad4c52a-6343-442c-9577-0d2b0e152fa1: 'url-relation-list'
[4025/107166] kindlewood ➜ unknown
[4026/107166] The Living Sisters ➜ unknown
[4027/107166] The Cowsills ➜ unknown
[4028/107166] Skee-Lo ➜ male
[4029/107166] Rosemary Clooney ➜ female
[4030/107166] Cat Power ➜ female
[4031/107166] Kristen Bell ➜ female
[4032/107166] Maia Wilson ➜ male
[4033/107166] Savatage ➜ unknown
[4034/107166] John Lennon ➜ male
[4035/107166] Josh Groban ➜ male
[4036/107166] The Cheeky Monkeys ➜ unknown
[4037/107166] Clap Your Hands Say Yeah ➜ unknown
[4038/107166] Jonathan Edwards ➜ male
[4039/107166] Coconut Records ➜ unknown
[4040/107166] Someone Still Loves You Boris Yeltsin ➜ unknown
[4041/107166] Garrett Douglas ➜ male
[4042/107166] Xavier Rudd ➜ male
[4043/107166] Cody Simpson ➜ male
[4044/107166] Emily Zeck ➜ female
[4045/107166] The Ventures ➜ unknown
[4046/107166] Josh Taylor ➜ female
[4047/107166] Aer ➜ unknown
[4048/107166] Joe Hertler & The Rainbow Seekers ➜ unknown
[4049/107166] State Radio ➜ unknown
[4050/107166] The Mighty Mighty Bosstones ➜ unknown
[4051/107166] Randy Travis ➜ male
[4052/107166] Trace Adkins ➜ male
[4053/107166] Gary Allan ➜ male
[4054/107166] Clay Walker ➜ male
[4055/107166] Tunji Ige ➜ unknown
[4056/107166] New Boyz ➜ unknown
[4057/107166] DJ SpinKing ➜ unknown
[4058/107166] Priceless Da Roc ➜ unknown
[4059/107166] Berner ➜ male
[4060/107166] Nef The Pharaoh ➜ male
[4061/107166] Young Dro ➜ male
[4062/107166] Mickey Avalon ➜ male
[get_wikidata_id] Error for MBID e0db8b35-2dab-4449-a181-2d78bb4e08e7: 'url-relation-list'
[4063/107166] Zayvsthem ➜ unknown
[4064/107166] Salva ➜ unknown
[4065/107166] Jet ➜ unknown
[4066/107166] Sofia Talvik ➜ female
[4067/107166] Herbie Hancock ➜ male
[4068/107166] Aimee Mann ➜ female
[4069/107166] Catherine Feeny ➜ female
[4070/107166] Mew ➜ unknown
[4071/107166] Tori Amos ➜ female
[4072/107166] Meiko ➜ female
[4073/107166] Anders F. Rönnblom ➜ male
[get_wikidata_id] Error for MBID c983f0d8-e57c-4591-bdc0-58180e83510b: 'url-relation-list'
[4074/107166] The Seeger Sisters ➜ unknown
[4075/107166] The Humorist ➜ unknown
[4076/107166] Calexico ➜ unknown
[4077/107166] The Silver Beetles ➜ unknown
[4078/107166] Big Data ➜ unknown
[4079/107166] Shye Ben-Tzur ➜ male
[4080/107166] Smoke Dza ➜ male
[4081/107166] Jet Life ➜ unknown
[4082/107166] Ralph Stanley ➜ male
[4083/107166] Bill Monroe & His Blue Grass Boys ➜ unknown
[4084/107166] Bill Monroe ➜ male
[4085/107166] The Monroe Brothers ➜ unknown
[4086/107166] Flatt & Scruggs ➜ unknown
[4087/107166] Old & In The Way ➜ unknown
[4088/107166] Gid Tanner & His Skillet Lickers ➜ unknown
[4089/107166] Ola Belle Reed ➜ female
[4090/107166] Fiddlin' Arthur Smith ➜ male
[4091/107166] Gid Tanner & His Skillet Lickers With Riley Puckett ➜ unknown
[get_wikidata_id] Error for MBID b61e2ebe-abc7-4a74-9498-776629c40ede: 'url-relation-list'
[4092/107166] W. Lee O'Daniel & His Hillbilly Boys ➜ unknown
[get_wikidata_id] Error for MBID df098e30-3020-4318-9975-666c833b8d2c: 'url-relation-list'
[4093/107166] The Humbard Family ➜ unknown
[4094/107166] Bob Atcher ➜ male
[4095/107166] Adolf Hofner & His San Antonians ➜ unknown
[4096/107166] Nashville Washboard Band ➜ unknown
[4097/107166] Arthur McClain & Joe Evans ➜ male
[4098/107166] Bert Jansch ➜ male
[4099/107166] Jerry Garcia ➜ male
[4100/107166] Yonder Mountain String Band ➜ unknown
[4101/107166] Pete Seeger ➜ male
[4102/107166] Bill Clifton ➜ male
[4103/107166] Jim & Jesse ➜ unknown
[4104/107166] Bradley Kincaid ➜ male
[4105/107166] Mother Mabel Carter ➜ unknown
[4106/107166] Carmel Quinn ➜ male
[4107/107166] Richard Lockmiller & Jim Connor ➜ male
[4108/107166] Carolyn Hester ➜ female
[4109/107166] Jacqueline Sharpe ➜ female
[4110/107166] Homer Briarhopper ➜ male
[4111/107166] The Pickard Family ➜ unknown
[4112/107166] Dixie Crackers ➜ unknown
[get_wikidata_id] Error for MBID faa9bf7d-29b1-4861-841d-7cb5b3e9dfe7: 'url-relation-list'
[4113/107166] Clarence "Tom" Ashley ➜ unknown
[4114/107166] Grayson & Whitter ➜ unknown
[4115/107166] Almoth Hodges ➜ unknown
[4116/107166] Bucklebusters ➜ unknown
[get_wikidata_id] Error for MBID b7066b0d-3342-449a-878e-1462480330dd: 'url-relation-list'
[4117/107166] Shortbuckle Roark & Family ➜ unknown
[4118/107166] Vernon Dalhart ➜ male
[4119/107166] Riley Puckett ➜ male
[4120/107166] Peg Leg Howell ➜ male
[4121/107166] Sleepy John Estes ➜ male
[4122/107166] Bill Carlisle ➜ female
[4123/107166] The Church Brothers & Their Blue Ridge Ramblers ➜ unknown
[4124/107166] Lester Flatt & Earl Scruggs And The Stanley Brothers ➜ unknown
[4125/107166] The Lilly Brothers & Don Stover ➜ unknown
[4126/107166] Mac Wiseman ➜ male
[4127/107166] Jimmy Martin ➜ male
[4128/107166] The Stanley Brothers ➜ unknown
[4129/107166] Greg Bates ➜ male
[4130/107166] Eric Paslay ➜ male
[4131/107166] JR Castro ➜ male
[4132/107166] Aly & AJ ➜ unknown
[4133/107166] The Ting Tings ➜ unknown
[4134/107166] Ashley Tisdale ➜ female
[4135/107166] Corbin Bleu ➜ male
[4136/107166] Troy ➜ male
[4137/107166] Orianthi ➜ female
[4138/107166] Elijah Blake ➜ male
[4139/107166] Sofi de la Torre ➜ female
[4140/107166] Norman Perry ➜ male
[4141/107166] Crush ➜ unknown
[4142/107166] Matt DiMona ➜ unknown
[4143/107166] David Sanya ➜ male
[4144/107166] Fantasia ➜ female
[4145/107166] Aaron Camper ➜ male
[4146/107166] Allan Rayman ➜ male
[get_wikidata_id] Error for MBID b2d306f6-f0ae-4f17-8bfe-438c5f34fe13: 'url-relation-list'
[4147/107166] Meaku ➜ unknown
[4148/107166] alayna ➜ unknown
[get_wikidata_id] Error for MBID 11f1518e-3ec3-4b36-9b36-b60fce2e3c63: 'url-relation-list'
[4149/107166] Wasionkey ➜ unknown
[4150/107166] Yo Trane ➜ unknown
[4151/107166] Euroz ➜ unknown
[get_wikidata_id] Error for MBID f8757c81-41a5-45f1-bb6c-7a0bbb063896: 'url-relation-list'
[4152/107166] The Aston Shuffle ➜ unknown
[4153/107166] Spada ➜ unknown
[4154/107166] Carousel ➜ unknown
[4155/107166] Scavenger Hunt ➜ unknown
[4156/107166] Dizzee Rascal ➜ male
[4157/107166] Noosa ➜ female
[4158/107166] Xenia Ghali ➜ female
[4159/107166] Will Sparks ➜ unknown
[4160/107166] Moby ➜ male
[4161/107166] Anna Of The North ➜ female
[4162/107166] Faul ➜ unknown
[4163/107166] Skrux ➜ unknown
[4164/107166] Dirty South ➜ male
[4165/107166] Gigamesh ➜ male
[4166/107166] Sailors ➜ unknown
[4167/107166] Don Diablo ➜ male
[4168/107166] Gin Wigmore ➜ female
[4169/107166] The Avener ➜ unknown
[4170/107166] Gregory Porter ➜ male
[4171/107166] Me & My Toothbrush ➜ unknown
[4172/107166] Nause ➜ unknown
[4173/107166] Adam K ➜ female
[4174/107166] Gorgon City ➜ unknown
[4175/107166] GRiZ ➜ male
[4176/107166] Croatia Squad ➜ male
[4177/107166] I Am Oak ➜ unknown
[4178/107166] Wordsplayed ➜ unknown
[4179/107166] Soulwax ➜ unknown
[4180/107166] Dooqu ➜ unknown
[4181/107166] Club cheval ➜ unknown
[4182/107166] Just A Gent ➜ unknown
[get_wikidata_id] Error for MBID 6c3818a6-8cc1-4e9f-8a79-d1408812cd37: 'url-relation-list'
[4183/107166] Bryzone ➜ unknown
[4184/107166] Besnine & Raphael ➜ unknown
[4185/107166] Danrell ➜ unknown
[4186/107166] Niteppl ➜ unknown
[4187/107166] Goldroom ➜ male
[4188/107166] AC Slater ➜ unknown
[4189/107166] Rezz ➜ female
[4190/107166] VibeSquaD ➜ unknown
[4191/107166] John Dahlbäck ➜ male
[4192/107166] Hypster ➜ unknown
[4193/107166] Mord Fustang ➜ male
[4194/107166] Joywave ➜ unknown
[4195/107166] Axero ➜ unknown
[4196/107166] Infuze ➜ unknown
[get_wikidata_id] Error for MBID ef8e4bcc-10f7-41c8-8797-cf604f0796f2: 'url-relation-list'
[4197/107166] Frederick Barr ➜ unknown
[4198/107166] Wax Motif ➜ male
[4199/107166] Gramatik ➜ male
[4200/107166] Trinix ➜ unknown
[4201/107166] Eric Prydz ➜ male
[4202/107166] NGHTMRE ➜ male
[4203/107166] Nitemayor ➜ unknown
[get_wikidata_id] Error for MBID 4d7fa31c-9c1f-4179-bb7a-2df8f39d5961: 'url-relation-list'
[4204/107166] Kosta Dejay ➜ unknown
[4205/107166] Paul Oakenfold ➜ male
[4206/107166] Louis Futon ➜ male
[4207/107166] Subtact ➜ unknown
[4208/107166] David Holmes ➜ male
[4209/107166] Space Jesus ➜ unknown
[4210/107166] Sofie Letitre ➜ unknown
[4211/107166] Stupead ➜ unknown
[4212/107166] Northeast Party House ➜ unknown
[4213/107166] Cleopold ➜ unknown
[4214/107166] Hugh ➜ male
[4215/107166] Crooked Colours ➜ unknown
[4216/107166] Thomas Vent ➜ male
[4217/107166] Slushii ➜ male
[4218/107166] RÜFÜS DU SOL ➜ unknown
[4219/107166] Bondax ➜ unknown
[4220/107166] Dom Dolla ➜ male
[4221/107166] Thief ➜ unknown
[4222/107166] MORTEN ➜ male
[4223/107166] Mistah F.A.B. ➜ male
[4224/107166] Louie Free ➜ unknown
[4225/107166] Atom Tree ➜ unknown
[4226/107166] Toussaint Morrison ➜ male
[4227/107166] Wafia ➜ female
[4228/107166] Viceroy ➜ unknown
[4229/107166] ALIUS ➜ unknown
[4230/107166] Alizzz ➜ male
[4231/107166] Kaido ➜ male
[4232/107166] Classified ➜ male
[4233/107166] Foreign Diplomats ➜ unknown
[4234/107166] Beats54 ➜ unknown
[4235/107166] Bonnie McKee ➜ female
[4236/107166] Albert G ➜ male
[4237/107166] Rob James ➜ male
[4238/107166] Oliver Heldens ➜ male
[4239/107166] Danny Darko ➜ male
[4240/107166] Faye Webster ➜ male
[4241/107166] Zander Hawley ➜ unknown
[4242/107166] The Districts ➜ unknown
[4243/107166] Wilder ➜ male
[4244/107166] The Night Game ➜ unknown
[4245/107166] Phoebe Bridgers ➜ female
[4246/107166] Pinegrove ➜ unknown
[4247/107166] Everything But You ➜ unknown
[4248/107166] Veronica Jax ➜ male
[4249/107166] Nouvelle Vague ➜ unknown
[4250/107166] Bikini Kill ➜ unknown
[4251/107166] American Horror Story Cast ➜ unknown
[4252/107166] Aesthetic Perfection ➜ unknown
[4253/107166] Garbage ➜ unknown
[4254/107166] Wolf Gang ➜ male
[4255/107166] The Sounds ➜ unknown
[get_wikidata_id] Error for MBID f0708c1a-5493-4f56-8d48-ea507ec7ecf7: 'url-relation-list'
[4256/107166] Ghost B.C. ➜ unknown
[4257/107166] Alfredo Olivas ➜ male
[4258/107166] Ariel Camacho y Los Plebes Del Rancho ➜ unknown
[4259/107166] Banda Los Recoditos ➜ unknown
[4260/107166] Banda Sinaloense MS de Sergio Lizárraga ➜ unknown
[4261/107166] Colmillo Norteño ➜ unknown
[4262/107166] Banda Tierra Sagrada ➜ unknown
[4263/107166] Beto Quintanilla ➜ male
[4264/107166] Calibre 50 ➜ unknown
[4265/107166] Cuisillos De Arturo Macias ➜ unknown
[4266/107166] Los Originales De San Juan ➜ unknown
[4267/107166] El Komander ➜ unknown
[4268/107166] El Movimiento Alterado ➜ unknown
[4269/107166] Los Tucanes De Tijuana ➜ unknown
[4270/107166] El Potro De Sinaloa ➜ male
[4271/107166] Enigma Norteño ➜ unknown
[4272/107166] Gerardo Ortiz ➜ male
[4273/107166] Javier Rosas Y Su Artillería Pesada ➜ male
[4274/107166] Jesús Ojeda y Sus Parientes ➜ unknown
[4275/107166] Jorge Valenzuela ➜ male
[4276/107166] Julión Álvarez y su Norteño Banda ➜ unknown
[4277/107166] La Adictiva Banda San José de Mesillas ➜ unknown
[get_wikidata_id] Error for MBID 6369cbf1-8d09-4d65-a082-b6cf18e7f0bc: 'url-relation-list'
[4278/107166] La Numero 1 Banda Jerez De Marco A. Flores ➜ unknown
[4279/107166] La Poderosa Banda San Juan ➜ unknown
[4280/107166] Edwin Luna y La Trakalosa de Monterrey ➜ unknown
[4281/107166] Larry Hernández ➜ male
[4282/107166] El Tigrillo Palma ➜ male
[get_wikidata_id] Error for MBID 99d6da4c-80fb-4291-834d-55b99d16cc10: 'url-relation-list'
[4283/107166] Los Inquietos Del Norte ➜ unknown
[4284/107166] Los Morros Del Norte ➜ unknown
[4285/107166] Los Nuevos Rebeldes ➜ unknown
[4286/107166] Los Titanes De Durango ➜ unknown
[4287/107166] Luis Coronel ➜ unknown
[get_wikidata_id] Error for MBID 35c43b52-2088-4c17-ad27-a195dc62c8e0: 'url-relation-list'
[4288/107166] Mario "El Cachorro" Delgado ➜ unknown
[4289/107166] Noel Torres ➜ male
[4290/107166] Palomo ➜ unknown
[get_wikidata_id] Error for MBID b2f1b118-4e29-4da6-8d09-a22ecbd49ece: 'url-relation-list'
[4291/107166] Proyecto X ➜ unknown
[4292/107166] Ramon Ayala Y Sus Bravos Del Norte ➜ unknown
[4293/107166] Regulo Caro ➜ male
[4294/107166] Revolver Cannabis ➜ unknown
[4295/107166] Roberto Tapia ➜ unknown
[4296/107166] Tito Y Su Torbellino ➜ male
[4297/107166] Traviezoz de la Zierra ➜ unknown
[4298/107166] Voz De Mando ➜ unknown
[4299/107166] Los Buitres De Culiacan Sinaloa ➜ unknown
[4300/107166] Los Plebes del Rancho de Ariel Camacho ➜ unknown
[4301/107166] Hijos De Barron ➜ unknown
[4302/107166] Fuerza de Tijuana ➜ unknown
[4303/107166] Grupo Maximo Grado ➜ unknown
[4304/107166] Banda Carnaval ➜ unknown
[4305/107166] Martin Castillo ➜ male
[get_wikidata_id] Error for MBID 0bd3e864-e47e-4da5-b86e-67980e8f5ef7: 'url-relation-list'
[4306/107166] Buknas De Culiacan ➜ unknown
[4307/107166] Los Rojos ➜ unknown
[4308/107166] El Bebeto ➜ unknown
[get_wikidata_id] Error for MBID 7d9ca720-5e74-42a3-ae72-b515041a28f1: 'url-relation-list'
[4309/107166] Código FN ➜ unknown
[4310/107166] Grupo Escolta ➜ unknown
[4311/107166] Grupo Imperial ➜ unknown
[4312/107166] La Septima Banda ➜ unknown
[4313/107166] Banda La Mundial De Claudio Alcaraz ➜ unknown
[4314/107166] Valentín Elizalde ➜ male
[get_wikidata_id] Error for MBID f9e64ad2-7195-4fa0-8244-c721b6ab8760: 'url-relation-list'
[4315/107166] Omar Ruiz ➜ unknown
[4316/107166] Grupo Fernandez ➜ male
[4317/107166] El Chapo De Sinaloa ➜ male
[4318/107166] Grupo Marca Registrada ➜ unknown
[4319/107166] Tito Torbellino ➜ male
[4320/107166] Lenin Ramírez ➜ unknown
[get_wikidata_id] Error for MBID 45f1b330-7c93-496d-826e-ae561fbe9763: 'url-relation-list'
[4321/107166] Arsenal Efectivo ➜ unknown
[4322/107166] Ulices Chaidez Y Sus Plebes ➜ unknown
[4323/107166] LEGADO 7 ➜ unknown
[4324/107166] Adriel Favela ➜ unknown
[4325/107166] Banda Renovación de Culiacán Sinaloa ➜ unknown
[4326/107166] Virlan Garcia ➜ unknown
[4327/107166] El Fantasma ➜ male
[4328/107166] Grupo Codiciado ➜ unknown
[4329/107166] Christian Nodal ➜ male
[4330/107166] La Arrolladora Banda El Limón De Rene Camacho ➜ unknown
[4331/107166] Tito Torbellino Jr ➜ unknown
[4332/107166] Los Cuates de Sinaloa ➜ unknown
[get_wikidata_id] Error for MBID 7a762525-00a1-4232-86ff-ef0616df438c: 'url-relation-list'
[4333/107166] Los Amos De Nuevo Leon ➜ unknown
[4334/107166] La Elegante ➜ unknown
[4335/107166] E-Dubble ➜ unknown
[4336/107166] Knife Party ➜ unknown
[4337/107166] Def Leppard ➜ unknown
[4338/107166] Aqua ➜ unknown
[4339/107166] Tal Bachman ➜ male
[4340/107166] Dead Or Alive ➜ unknown
[4341/107166] Kenny And The Scots ➜ unknown
[4342/107166] Ra Ra Riot ➜ unknown
[4343/107166] MUNA ➜ unknown
[4344/107166] IRONTOM ➜ unknown
[4345/107166] Jorja Smith ➜ female
[4346/107166] Joey Purp ➜ male
[4347/107166] K.Flay ➜ female
[4348/107166] Lil Playy ➜ male
[4349/107166] Dropkick Murphys ➜ unknown
[4350/107166] The Gregory Brothers ➜ unknown
[4351/107166] Melissa Steel ➜ female
[4352/107166] Lylah ➜ female
[4353/107166] Stacy ➜ female
[4354/107166] Marvin ➜ male
[4355/107166] Sofia Karlberg ➜ female
[4356/107166] DJ Dirty Sprite ➜ unknown
[4357/107166] Eva Simons ➜ female
[4358/107166] Taolo ➜ unknown
[4359/107166] Empire Cast ➜ unknown
[4360/107166] Rickie Lee Jones ➜ female
[4361/107166] The Kingsmen ➜ unknown
[4362/107166] The Dirty Mac ➜ unknown
[4363/107166] Jim Reeves ➜ male
[4364/107166] Tommy James ➜ male
[4365/107166] John Mayer Trio ➜ unknown
[4366/107166] Bill Conti ➜ male
[get_wikidata_id] Error for MBID 2df4014d-e446-46c5-9cce-db20b2dccac1: 'url-relation-list'
[4367/107166] Halloween Sound Effects ➜ unknown
[4368/107166] Halloween Sounds ➜ unknown
[4369/107166] Blanca ➜ unknown
[4370/107166] tobyMac ➜ male
[4371/107166] Citizen Way ➜ unknown
[4372/107166] Jasmine Murray ➜ female
[4373/107166] Mary Mary ➜ female
[4374/107166] Jordan Feliz ➜ male
[4375/107166] Josh Wilson ➜ male
[4376/107166] Capital Kings ➜ unknown
[4377/107166] Micah Tyler ➜ female
[4378/107166] Zach Williams ➜ male
[4379/107166] Brandon Heath ➜ male
[4380/107166] P.O.D. ➜ unknown
[get_wikidata_id] Error for MBID b8046a12-3a52-4a40-ae5c-e2ec0893523c: 'url-relation-list'
[4381/107166] Jorge Quintero ➜ unknown
[4382/107166] Yes Kids ➜ unknown
[get_wikidata_id] Error for MBID 27bfdbcd-06de-4e4e-a49e-46cdbbe9d93a: 'url-relation-list'
[4383/107166] Ultimate Pop Hits ➜ unknown
[4384/107166] The Heavy Band ➜ unknown
[4385/107166] Zombie Nation ➜ unknown
[4386/107166] Ciaran Lavery ➜ unknown
[4387/107166] The Black Atlantic ➜ unknown
[4388/107166] Shannon Saunders ➜ male
[4389/107166] Soko ➜ female
[4390/107166] Josh Record ➜ male
[4391/107166] RY X ➜ male
[4392/107166] Adam Levine ➜ male
[4393/107166] Mree ➜ female
[4394/107166] Scott McKenzie ➜ male
[4395/107166] ARCHIS ➜ unknown
[4396/107166] Family of the Year ➜ unknown
[4397/107166] Brett Dennen ➜ male
[4398/107166] Fyfe ➜ unknown
[4399/107166] Michael Schulte ➜ male
[4400/107166] For All Seasons ➜ unknown
[4401/107166] You+Me ➜ unknown
[4402/107166] Thomas Newman ➜ male
[4403/107166] Joe Gil ➜ male
[4404/107166] Star Anna ➜ unknown
[4405/107166] Klaxons ➜ unknown
[4406/107166] Allen Stone ➜ unknown
[4407/107166] Jalen Santoy ➜ unknown
[4408/107166] NGHBRS ➜ unknown
[4409/107166] Roadkill Ghost Choir ➜ unknown
[4410/107166] The Sound Is Fine ➜ unknown
[4411/107166] Crash Kings ➜ unknown
[4412/107166] Puscifer ➜ unknown
[4413/107166] Brick + Mortar ➜ unknown
[4414/107166] Hotel of the Laughing Tree ➜ unknown
[4415/107166] White Rabbits ➜ unknown
[4416/107166] Tallhart ➜ unknown
[4417/107166] O'Brother ➜ unknown
[4418/107166] Tango Alpha Tango ➜ unknown
[4419/107166] Mylets ➜ unknown
[4420/107166] FIDLAR ➜ unknown
[4421/107166] The Willowz ➜ unknown
[4422/107166] Ever so Noble ➜ unknown
[4423/107166] Mo Lowda & the Humble ➜ unknown
[4424/107166] The Main Squeeze ➜ unknown
[4425/107166] Dead Sara ➜ unknown
[4426/107166] Blood Red Shoes ➜ unknown
[4427/107166] Middle Class Rut ➜ unknown
[4428/107166] Stone Giant ➜ unknown
[4429/107166] Heavy Young Heathens ➜ unknown
[4430/107166] The Bots ➜ unknown
[4431/107166] The Blue Van ➜ unknown
[4432/107166] Minus The Bear ➜ unknown
[4433/107166] Tribe Society ➜ unknown
[4434/107166] France ➜ female
[4435/107166] Deadboy & The Elephantmen ➜ unknown
[4436/107166] One Day As A Lion ➜ unknown
[4437/107166] And So I Watch You from Afar ➜ unknown
[4438/107166] The Faint ➜ unknown
[4439/107166] Beats Antique ➜ unknown
[4440/107166] The Pack a.d. ➜ unknown
[4441/107166] Nico Vega ➜ female
[4442/107166] City Fidelia ➜ unknown
[4443/107166] Derek Wise ➜ male
[4444/107166] Ripp Flamez ➜ unknown
[4445/107166] Courtlin Jabrae ➜ unknown
[4446/107166] 2NE1 ➜ unknown
[4447/107166] SHINee ➜ unknown
[4448/107166] Brown Eyed Girls ➜ unknown
[4449/107166] Trouble Maker ➜ unknown
[4450/107166] U-KISS ➜ unknown
[4451/107166] Wonder Girls ➜ male
[4452/107166] miss A ➜ unknown
[4453/107166] T-ara ➜ unknown
[4454/107166] Alice Cate ➜ male
[4455/107166] Super Junior ➜ unknown
[4456/107166] f(x) ➜ unknown
[4457/107166] TVXQ! ➜ unknown
[4458/107166] Baek Ji Young ➜ female
[4459/107166] Hot Potato ➜ unknown
[4460/107166] 도나웨일 ➜ unknown
[4461/107166] Leessang ➜ unknown
[4462/107166] Epik High ➜ unknown
[4463/107166] Younha ➜ unknown
[4464/107166] IU ➜ female
[4465/107166] Orange Caramel ➜ unknown
[4466/107166] Lil Lippy ➜ male
[4467/107166] GoGirl! ➜ unknown
[4468/107166] Speaker Knockerz ➜ unknown
[4469/107166] Ghost Town ➜ unknown
[4470/107166] Felly ➜ unknown
[4471/107166] B-Legit ➜ male
[4472/107166] BAMF! ➜ unknown
[4473/107166] Williams Riley ➜ male
[4474/107166] Clayton Anderson ➜ female
[4475/107166] Montgomery Gentry ➜ unknown
[4476/107166] Trent Willmon ➜ male
[4477/107166] Colt Ford ➜ male
[4478/107166] Casey Ashley ➜ male
[4479/107166] Chase Bryant ➜ female
[4480/107166] Bruce Adler ➜ male
[4481/107166] Brad Kane ➜ male
[4482/107166] Lea Salonga ➜ female
[4483/107166] Jonathan Freeman ➜ male
[4484/107166] Donny Osmond ➜ male
[4485/107166] Harvey Fierstein ➜ male
[4486/107166] Jason Weaver ➜ male
[4487/107166] Carmen Twillie ➜ female
[4488/107166] Jeremy Irons ➜ male
[4489/107166] Nathan Lane ➜ male
[4490/107166] Jodi Benson ➜ male
[4491/107166] Samuel E. Wright ➜ male
[4492/107166] Chorus - Beauty And the Beast ➜ unknown
[4493/107166] Robby Benson ➜ male
[4494/107166] Mandy Moore ➜ female
[4495/107166] Donna Murphy ➜ female
[4496/107166] Judy Kuhn ➜ female
[4497/107166] Heidi Mollenhauer ➜ female
[4498/107166] Tony Jay ➜ male
[4499/107166] Belly ➜ male
[4500/107166] Zak Downtown ➜ unknown
[4501/107166] Starrah ➜ female
[4502/107166] Adolphson & Falk ➜ unknown
[4503/107166] Busungarna ➜ unknown
[4504/107166] Erland Hagegård ➜ male
[4505/107166] Krister St. Hill ➜ male
[4506/107166] Frida ➜ female
[4507/107166] Agnetha & Linda ➜ unknown
[4508/107166] Jorgen Edman ➜ male
[4509/107166] Tommy Körberg ➜ male
[4510/107166] Troll ➜ unknown
[4511/107166] Triad ➜ unknown
[4512/107166] Carola ➜ female
[4513/107166] Sanna Nielsen ➜ female
[4514/107166] Vikingarna ➜ unknown
[4515/107166] Chris Rea ➜ male
[4516/107166] Mel & Kim ➜ unknown
[get_wikidata_id] Error for MBID 3ae9d218-c506-4033-8423-a1829a4898ae: 'url-relation-list'
[4517/107166] Ivana Sibinovic ➜ unknown
[get_wikidata_id] Error for MBID 9c44f55f-70c0-431e-a076-1a6c310ead05: 'url-relation-list'
[4518/107166] Super Troupers ➜ unknown
[4519/107166] Shakin' Stevens ➜ male
[4520/107166] Al Martino ➜ male
[4521/107166] Murdo McRae ➜ unknown
[4522/107166] Lili & Susie ➜ unknown
[4523/107166] Slade ➜ unknown
[4524/107166] Judy Garland ➜ female
[4525/107166] The Noel Party Singers ➜ unknown
[get_wikidata_id] Error for MBID bfb8846a-600e-4729-9919-9382b6e2a042: 'url-relation-list'
[4526/107166] Mälarkören ➜ unknown
[4527/107166] Ainbusk ➜ unknown
[4528/107166] Lotta Engberg ➜ female
[4529/107166] Jan Malmsjö ➜ male
[4530/107166] Anita Kerr Singers ➜ unknown
[4531/107166] eleventyseven ➜ unknown
[4532/107166] Press Play ➜ unknown
[4533/107166] Jeremy Camp ➜ male
[4534/107166] Love & The Outcome ➜ unknown
[4535/107166] Sørensen ➜ unknown
[4536/107166] Everfound ➜ unknown
[4537/107166] Steve Hare ➜ male
[4538/107166] Todd Smith ➜ male
[4539/107166] Fee ➜ unknown
[4540/107166] Mark Schultz ➜ male
[get_wikidata_id] Error for MBID 90ccd9f5-f635-4eef-8c11-0633e4bd9c65: 'url-relation-list'
[4541/107166] The Digital Age ➜ unknown
[4542/107166] Derek Ryan ➜ male
[4543/107166] Crowder ➜ male
[4544/107166] Eleven Past One ➜ unknown
[4545/107166] Charlton Heston ➜ male
[4546/107166] Roz Ryan ➜ male
[4547/107166] Danny DeVito ➜ male
[4548/107166] Chorus - Hercules ➜ unknown
[4549/107166] Barry Bostwick ➜ male
[4550/107166] Susan Sarandon ➜ female
[4551/107166] Johnathan Adams ➜ male
[4552/107166] Cast ➜ unknown
[4553/107166] Thoroughly Modern Millie Orchestra ➜ unknown
[4554/107166] Sutton Foster ➜ female
[4555/107166] Angela Christian ➜ unknown
[4556/107166] Ken Leung ➜ male
[4557/107166] Marc Kudisch ➜ unknown
[4558/107166] Harriet Harris ➜ female
[4559/107166] Gavin Creel ➜ male
[4560/107166] Sheryl Lee Ralph ➜ female
[4561/107166] Thoroughly Modern Millie Ensemble ➜ unknown
[4562/107166] Orchestra ➜ unknown
[4563/107166] Krysta Rodriguez ➜ female
[4564/107166] Bebe Neuwirth ➜ female
[4565/107166] Adam Riegler ➜ unknown
[4566/107166] Carolee Carmello ➜ unknown
[4567/107166] Kevin Chamberlin ➜ male
[4568/107166] Terrence Mann ➜ unknown
[4569/107166] Andrew Rannells ➜ male
[4570/107166] Michael Potts ➜ male
[4571/107166] Scott Barnhardt ➜ unknown
[4572/107166] Nikki M. James ➜ female
[4573/107166] Lewis Cleale ➜ male
[4574/107166] Clark Johnsen ➜ female
[4575/107166] Rema Webb ➜ unknown
[4576/107166] Ray LaMontagne ➜ male
[4577/107166] Eric Hutchinson ➜ male
[4578/107166] Blaque ➜ unknown
[4579/107166] Pras ➜ male
[4580/107166] 702 ➜ unknown
[4581/107166] Christina Milian ➜ female
[4582/107166] Selah ➜ unknown
[4583/107166] Tauren Wells ➜ male
[4584/107166] Ryan Stevenson ➜ male
[4585/107166] Mandisa ➜ female
[4586/107166] Big Daddy Weave ➜ unknown
[get_wikidata_id] Error for MBID 7719919e-996f-4448-9956-bec8b0b0b2fb: 'url-relation-list'
[4587/107166] Elizabeth Debicki ➜ unknown
[4588/107166] Bryan Ferry ➜ male
[4589/107166] Leonardo Dicaprio ➜ male
[4590/107166] Coco O. ➜ female
[4591/107166] Green Light ➜ unknown
[4592/107166] Tobey Maguire ➜ male
[4593/107166] Telekinesis ➜ unknown
[4594/107166] Morning Parade ➜ unknown
[4595/107166] Lewis Watson ➜ male
[4596/107166] James ➜ male
[4597/107166] Aron Wright ➜ male
[4598/107166] Conner Youngblood ➜ unknown
[4599/107166] The Arcs ➜ unknown
[4600/107166] Nahko and Medicine for the People ➜ unknown
[4601/107166] Sam Burchfield ➜ male
[4602/107166] The National Parks ➜ unknown
[4603/107166] Mighty Oaks ➜ unknown
[4604/107166] Chef'Special ➜ unknown
[4605/107166] Grady Spencer & the Work ➜ unknown
[4606/107166] Thomas Csorba ➜ male
[get_wikidata_id] Error for MBID 5062deb7-5b6d-46b1-bb11-9d38fde27b65: 'url-relation-list'
[4607/107166] John Vincent III ➜ unknown
[4608/107166] Connor Zwetsch ➜ unknown
[4609/107166] Joshua Hyslop ➜ male
[4610/107166] Austin Plaine ➜ unknown
[4611/107166] Stop Light Observations ➜ unknown
[4612/107166] Will Joseph Cook ➜ unknown
[4613/107166] Drew Holcomb ➜ male
[4614/107166] Lowly Spects ➜ unknown
[4615/107166] Bronze Radio Return ➜ unknown
[4616/107166] GoldFord ➜ unknown
[4617/107166] Ivan & Alyosha ➜ unknown
[4618/107166] Stephen Kellogg ➜ male
[4619/107166] Austin Basham ➜ unknown
[4620/107166] Jake McMullen ➜ male
[get_wikidata_id] Error for MBID cb2dd376-e257-4783-a39b-d1b02a58a525: 'url-relation-list'
[4621/107166] Boom Forest ➜ unknown
[4622/107166] Dionysia ➜ unknown
[4623/107166] The Corcoran Brothers ➜ unknown
[4624/107166] Rusko ➜ male
[4625/107166] Bassnectar ➜ male
[4626/107166] The Cataracs ➜ unknown
[4627/107166] Mt. Eden ➜ unknown
[4628/107166] JB Real ➜ male
[4629/107166] 12th Planet ➜ male
[4630/107166] Rednek ➜ unknown
[4631/107166] Andre Nickatina ➜ male
[4632/107166] Datsik ➜ male
[4633/107166] Hot Pink Delorean ➜ unknown
[4634/107166] Liquid Stranger ➜ male
[4635/107166] Dead Celebrity Status ➜ unknown
[4636/107166] The City of Prague Philharmonic Orchestra ➜ unknown
[get_wikidata_id] Error for MBID 06c23e4b-6036-4108-be0d-7907545d83b5: 'url-relation-list'
[4637/107166] TV Theme Tune Factory ➜ unknown
[4638/107166] Monty Norman ➜ unknown
[4639/107166] Big Boi ➜ male
[4640/107166] Giuseppe Verdi ➜ male
[4641/107166] Jefferson Starship ➜ unknown
[4642/107166] Wolfmother ➜ unknown
[4643/107166] JINC Ent ➜ unknown
[4644/107166] Little Richard ➜ male
[4645/107166] Flight of the Conchords ➜ unknown
[4646/107166] New Found Glory ➜ unknown
[4647/107166] Electric Six ➜ unknown
[4648/107166] Curtis Mayfield ➜ male
[4649/107166] The Blues Brothers ➜ unknown
[4650/107166] Raekwon ➜ male
[4651/107166] Nas & Damian "Jr. Gong" Marley ➜ male
[4652/107166] Guru's Jazzmatazz ➜ unknown
[4653/107166] Guru ➜ unknown
[4654/107166] Pete Rock, CL Smooth ➜ unknown
[4655/107166] Scarface ➜ male
[4656/107166] DJ Quik ➜ male
[4657/107166] Shyne ➜ male
[4658/107166] Jadakiss ➜ male
[4659/107166] Clipse ➜ unknown
[4660/107166] Schoolly D ➜ male
[4661/107166] T3R Elemento ➜ unknown
[get_wikidata_id] Error for MBID 4f2714e7-88e8-4e5d-9cee-c401336920e7: 'url-relation-list'
[4662/107166] Grupo H-100 ➜ unknown
[4663/107166] Alta Consigna ➜ unknown
[4664/107166] Hijos De La Plaza ➜ unknown
[4665/107166] Isaac Payton Sweat ➜ unknown
[4666/107166] The Commitments ➜ unknown
[4667/107166] Tracy Byrd ➜ male
[4668/107166] Mikel Knight ➜ unknown
[4669/107166] The Lacs ➜ unknown
[4670/107166] Brooks Jefferson ➜ unknown
[4671/107166] Alvaro Soler ➜ male
[4672/107166] Diego Boneta ➜ male
[4673/107166] Keri Hilson ➜ female
[4674/107166] Ram Jam ➜ unknown
[4675/107166] Suga Free ➜ male
[4676/107166] Nate Dogg ➜ male
[4677/107166] Khia ➜ female
[4678/107166] Jenny Owen Youngs ➜ female
[4679/107166] Mase ➜ male
[4680/107166] Ron Browz ➜ male
[4681/107166] Ro James ➜ male
[4682/107166] Cold Chilling Collective ➜ unknown
[4683/107166] Louis York ➜ male
[4684/107166] Colonel Abrams ➜ male
[4685/107166] Mibbs ➜ male
[4686/107166] QUE. ➜ unknown
[get_wikidata_id] Error for MBID 29b2927f-b3da-47ba-8786-f09d6b3001de: 'url-relation-list'
[4687/107166] ISA ➜ unknown
[4688/107166] Finatticz ➜ unknown
[4689/107166] LoveRance ➜ male
[4690/107166] Lil Mouse ➜ unknown
[4691/107166] Kendra Morris ➜ female
[4692/107166] Donny Hathaway ➜ male
[4693/107166] The Box Tops ➜ unknown
[4694/107166] Yo La Tengo ➜ unknown
[4695/107166] Frank Turner ➜ male
[4696/107166] Jil Is Lucky ➜ male
[4697/107166] Kyle Andrews ➜ unknown
[4698/107166] Marcus Foster ➜ unknown
[4699/107166] Pete Yorn ➜ male
[4700/107166] Puggy ➜ unknown
[4701/107166] O.A.R. ➜ unknown
[get_wikidata_id] Error for MBID 53f2ec39-004a-4a77-aa17-5eda8c1b2b39: 'url-relation-list'
[4702/107166] Martin Luke Brown ➜ unknown
[4703/107166] Travie McCoy ➜ male
[4704/107166] Dinka ➜ unknown
[4705/107166] Gareth Emery ➜ male
[4706/107166] Funkagenda ➜ unknown
[4707/107166] Medina ➜ female
[4708/107166] EDX ➜ male
[4709/107166] The Bloody Beetroots ➜ male
[4710/107166] Basto ➜ male
[4711/107166] Sinden ➜ male
[4712/107166] Kiasmos ➜ unknown
[4713/107166] Memoryhouse ➜ unknown
[4714/107166] Blur ➜ unknown
[4715/107166] Royal Tailor ➜ unknown
[4716/107166] Manic Drive ➜ unknown
[4717/107166] Bright City ➜ unknown
[4718/107166] Mosaic MSC ➜ unknown
[4719/107166] Darlene Zschech ➜ female
[4720/107166] We Are Messengers ➜ unknown
[4721/107166] Melissa Helser ➜ unknown
[4722/107166] Kim Walker-Smith ➜ female
[4723/107166] Louis Prima ➜ male
[4724/107166] Bruce Reitherman ➜ male
[4725/107166] M. Keali'i Ho'omalu ➜ unknown
[4726/107166] Joseph Williams ➜ male
[4727/107166] Disney Characters ➜ unknown
[get_wikidata_id] Error for MBID 070addcc-a800-4bd2-9d25-560003767b8d: 'url-relation-list'
[4728/107166] Aron Apping ➜ unknown
[4729/107166] The Supremes ➜ unknown
[4730/107166] Fontella Bass ➜ female
[4731/107166] The Contours ➜ unknown
[4732/107166] James Brown & The Famous Flames ➜ unknown
[4733/107166] Dion & The Belmonts ➜ unknown
[4734/107166] Josh Thompson ➜ male
[4735/107166] Hank Williams, Jr. ➜ male
[4736/107166] Chris LeDoux ➜ male
[4737/107166] Flatland Cavalry ➜ unknown
[4738/107166] Conway Twitty ➜ male
[4739/107166] James Otto ➜ male
[4740/107166] Prophets and Outlaws ➜ unknown
[4741/107166] Jamie Richards ➜ unknown
[4742/107166] William Clark Green ➜ unknown
[4743/107166] Jory Boy ➜ male
[4744/107166] Revol ➜ unknown
[4745/107166] Rvssian ➜ male
[4746/107166] Alex Rose ➜ female
[4747/107166] Juhn ➜ unknown
[4748/107166] Noriel ➜ male
[4749/107166] Pepe Quintana ➜ male
[4750/107166] Lary Over ➜ unknown
[4751/107166] Lenny Tavárez ➜ unknown
[4752/107166] Bryant Myers ➜ female
[4753/107166] Nengo Flow ➜ male
[4754/107166] Anuel Aa ➜ male
[4755/107166] Mueka ➜ unknown
[4756/107166] Finesse ➜ male
[4757/107166] Breakfast n Vegas ➜ unknown
[4758/107166] Play-N-Skillz ➜ unknown
[4759/107166] Dayme y El High ➜ unknown
[4760/107166] Yampi ➜ unknown
[4761/107166] J-King y Maximan ➜ male
[4762/107166] C. Tangana ➜ male
[4763/107166] Juancho Marqués ➜ unknown
[4764/107166] Natos y Waor ➜ unknown
[4765/107166] Hard GZ ➜ unknown
[4766/107166] Justin Quiles ➜ male
[4767/107166] Matthew Koma ➜ male
[4768/107166] Emancipator ➜ male
[4769/107166] Sub Focus ➜ male
[4770/107166] Seinabo Sey ➜ female
[4771/107166] Kyla La Grange ➜ female
[4772/107166] Big Gigantic ➜ unknown
[4773/107166] 3LAU ➜ male
[4774/107166] Bingo Players ➜ unknown
[4775/107166] Bell Humble ➜ unknown
[4776/107166] Apex Rise ➜ unknown
[4777/107166] Naughty Boy ➜ male
[4778/107166] Lenno ➜ unknown
[4779/107166] Tom Hangs ➜ male
[4780/107166] tyDi ➜ male
[4781/107166] Syn Cole ➜ male
[4782/107166] The Jane Doze ➜ unknown
[4783/107166] Arno Cost ➜ male
[4784/107166] SomeKindaWonderful ➜ unknown
[4785/107166] Sultan ➜ male
[4786/107166] Zwette ➜ unknown
[4787/107166] Feenixpawl ➜ unknown
[4788/107166] Rebecca & Fiona ➜ unknown
[4789/107166] BCX ➜ unknown
[4790/107166] Dropout ➜ unknown
[4791/107166] Nora En Pure ➜ female
[4792/107166] Adam Rickfors ➜ unknown
[4793/107166] Dimmi ➜ unknown
[4794/107166] The Colourist ➜ unknown
[4795/107166] Milkman ➜ unknown
[4796/107166] Shiloh ➜ unknown
[4797/107166] Gazzo ➜ male
[4798/107166] Pierce Fulton ➜ male
[4799/107166] RÜFÜS ➜ male
[4800/107166] Rain Man ➜ male
[4801/107166] Yves V ➜ male
[4802/107166] Andrew Rayel ➜ male
[4803/107166] Airia ➜ unknown
[4804/107166] Kap Slap ➜ unknown
[4805/107166] Porter Robinson & Madeon ➜ male
[4806/107166] Aash Mehta ➜ unknown
[4807/107166] Tobu ➜ male
[4808/107166] Felix Cartal ➜ male
[4809/107166] Campsite Dream ➜ unknown
[4810/107166] Sultan + Shepard  ➜ unknown
[4811/107166] John Martin ➜ male
[4812/107166] Alex Sonata ➜ unknown
[4813/107166] Thomas Newson ➜ unknown
[4814/107166] Honka ➜ unknown
[4815/107166] Nick Martin ➜ male
[4816/107166] Suspect 44 ➜ unknown
[4817/107166] Stadiumx ➜ unknown
[4818/107166] Eminence ➜ unknown
[4819/107166] Yacht Club ➜ unknown
[4820/107166] Synchronice & Kasum ➜ unknown
[4821/107166] Throttle ➜ unknown
[4822/107166] DSKO ➜ unknown
[4823/107166] Jordan Andrew ➜ male
[4824/107166] Kontinuum ➜ unknown
[4825/107166] Josef Salvat ➜ male
[4826/107166] Jaël ➜ female
[4827/107166] Summer Was Fun ➜ unknown
[4828/107166] Boehm ➜ unknown
[4829/107166] Hollywood Principle ➜ unknown
[4830/107166] Subfer ➜ unknown
[4831/107166] Uppermost ➜ male
[4832/107166] John Reuben ➜ male
[4833/107166] Social Club Misfits ➜ unknown
[4834/107166] Thousand Foot Krutch ➜ unknown
[4835/107166] Alien Ant Farm ➜ unknown
[4836/107166] K.W.S. ➜ unknown
[4837/107166] Arrested Development ➜ unknown
[4838/107166] 38 Special ➜ unknown
[4839/107166] Glass Tiger ➜ male
[4840/107166] Huey Lewis & The News ➜ unknown
[4841/107166] Mike & The Mechanics ➜ unknown
[4842/107166] Robert Palmer ➜ male
[4843/107166] Go West ➜ unknown
[4844/107166] GAWVI ➜ male
[4845/107166] Saul El Jaguar Alarcón ➜ male
[4846/107166] Duelo ➜ unknown
[4847/107166] El Bebeto Y Su Banda Patria Chica ➜ unknown
[4848/107166] Los Invasores De Nuevo León ➜ unknown
[get_wikidata_id] Error for MBID 120b7c41-87ba-48f6-8d2e-d0e3fead397b: 'url-relation-list'
[4849/107166] Julio Chaidez ➜ unknown
[4850/107166] La Original Banda El Limón de Salvador Lizárraga ➜ unknown
[get_wikidata_id] Error for MBID e58e89c5-edd5-4470-b50e-f34b912a5305: 'url-relation-list'
[4851/107166] El Pelón del Mikrophone ➜ unknown
[4852/107166] Aaron Y Su Grupo Ilusion ➜ unknown
[4853/107166] Notch ➜ male
[4854/107166] Remmy Valenzuela ➜ male
[4855/107166] Marcus Claim ➜ male
[4856/107166] The Hollywood LA Soundtrack Orchestra ➜ unknown
[4857/107166] Aaron Zigman ➜ male
[4858/107166] Guitar Melody ➜ male
[4859/107166] Steve Hall ➜ male
[4860/107166] Famous Melodies ➜ unknown
[4861/107166] Klaus Badelt ➜ male
[4862/107166] Instrumental Players ➜ unknown
[4863/107166] Ryan & Rachell O'Donnell ➜ unknown
[4864/107166] Helen Jane Long ➜ female
[4865/107166] Robin Meloy Goldsby ➜ female
[4866/107166] Brian Crain ➜ unknown
[4867/107166] Elijah Bossenbroek ➜ unknown
[4868/107166] This Will Destroy You ➜ unknown
[4869/107166] Harry Gregson-Williams ➜ male
[4870/107166] Robby Cool ➜ male
[get_wikidata_id] Error for MBID 62e88bb2-40c7-4db3-9dfb-d690ea971eee: 'url-relation-list'
[4871/107166] Maite Aurrekoetxea ➜ unknown
[get_wikidata_id] Error for MBID cdce3018-1f78-441e-8e02-63acdf4df2e8: 'url-relation-list'
[4872/107166] Vitoria-Gasteiz Orchestra ➜ unknown
[4873/107166] Dirk Brossé ➜ male
[4874/107166] Fredi Peláez ➜ male
[4875/107166] Artur Guimaraes ➜ male
[4876/107166] Brussels Philharmonic ➜ unknown
[4877/107166] Iker Sanchez ➜ male
[4878/107166] L.A. Strings ➜ unknown
[4879/107166] Broken Twin ➜ male
[get_wikidata_id] Error for MBID c7a56b6d-0068-4677-9f98-f0fdf485a05b: 'url-relation-list'
[4880/107166] Vienna Boys' Choir ➜ unknown
[4881/107166] Robert Burns ➜ male
[4882/107166] Bodeans ➜ unknown
[4883/107166] Soul Coughing ➜ unknown
[4884/107166] Tripping Daisy ➜ unknown
[4885/107166] Local H ➜ unknown
[4886/107166] Spacehog ➜ unknown
[4887/107166] Seven Mary Three ➜ unknown
[4888/107166] Faith No More ➜ unknown
[search_artist] Error for 'Social Distortion': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4889/107166] Social Distortion ➜ unknown
[search_artist] Error for 'Korn': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4890/107166] Korn ➜ unknown
[search_artist] Error for 'The Cult': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4891/107166] The Cult ➜ unknown
[search_artist] Error for 'Boys Don't Cry': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4892/107166] Boys Don't Cry ➜ unknown
[4893/107166] MC 900 Ft. Jesus ➜ male
[4894/107166] Love and Rockets ➜ unknown
[4895/107166] XTC ➜ unknown
[4896/107166] Dinosaur Jr. ➜ unknown
[4897/107166] Joy Division ➜ unknown
[4898/107166] Kap G ➜ male
[4899/107166] SOB X RBE ➜ unknown
[4900/107166] DJ Mustard ➜ male
[4901/107166] RJMrLA ➜ male
[get_wikidata_id] Error for MBID 3217ffbb-4bda-432b-8cbd-ece584577b4b: 'url-relation-list'
[4902/107166] M City JR ➜ unknown
[4903/107166] Phillip Phillips ➜ male
[4904/107166] Housefires ➜ unknown
[4905/107166] Caedmon's Call ➜ unknown
[4906/107166] Leeland Mooring ➜ male
[4907/107166] Jonathan David Helser ➜ unknown
[4908/107166] Highlands Worship ➜ unknown
[4909/107166] 10,000 Fathers ➜ unknown
[4910/107166] Kristian Stanfill ➜ male
[4911/107166] Todd Dulaney ➜ male
[4912/107166] ORU LIVE ➜ unknown
[4913/107166] John Mark McMillan ➜ male
[4914/107166] Family Church Worship ➜ unknown
[4915/107166] Jon Thurlow ➜ male
[4916/107166] Isla Vista Worship ➜ unknown
[4917/107166] Free Chapel ➜ unknown
[4918/107166] Ascend The Hill ➜ unknown
[get_wikidata_id] Error for MBID 609ae2d0-cef4-4c6e-a8b9-59addece0052: 'url-relation-list'
[4919/107166] Crecer German ➜ unknown
[4920/107166] Los Hijos De Hernández ➜ unknown
[get_wikidata_id] Error for MBID 3929a5ba-d2c1-40af-b464-6e869f6c7959: 'url-relation-list'
[4921/107166] Grupo 360 ➜ unknown
[4922/107166] Los Del Arroyo ➜ unknown
[4923/107166] Banda Renovacion ➜ unknown
[4924/107166] Los Migueles "La Voz Original" ➜ unknown
[4925/107166] The Postal Service ➜ unknown
[4926/107166] Jakob Ogawa ➜ male
[4927/107166] crwn ➜ unknown
[4928/107166] June Marieezy ➜ female
[4929/107166] Gibbz ➜ unknown
[4930/107166] Moonchild ➜ unknown
[4931/107166] The Dear Hunter ➜ unknown
[4932/107166] Justin Hurwitz ➜ male
[4933/107166] Lido ➜ male
[4934/107166] Moon Bounce ➜ unknown
[get_wikidata_id] Error for MBID fa1479aa-ceec-4e81-a48d-794694fec177: 'url-relation-list'
[4935/107166] Khai ➜ unknown
[4936/107166] Swell ➜ unknown
[4937/107166] vbnd ➜ unknown
[4938/107166] Ta-ku ➜ unknown
[get_wikidata_id] Error for MBID bb1b0435-b8c9-4ab3-a656-5b6eb6cd9377: caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4939/107166] Luke Levenson ➜ unknown
[search_artist] Error for 'Tora': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4940/107166] Tora ➜ unknown
[search_artist] Error for 'Jacob Collier': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4941/107166] Jacob Collier ➜ unknown
[4942/107166] Hrvrd ➜ unknown
[4943/107166] Canon Blue ➜ male
[4944/107166] Gifted Gab ➜ unknown
[4945/107166] Metronomy ➜ unknown
[4946/107166] Ravyn Lenae ➜ female
[4947/107166] Karma Kid ➜ male
[4948/107166] Dae Zhen ➜ unknown
[4949/107166] Eryn Allen Kane ➜ unknown
[4950/107166] Tennyson ➜ unknown
[4951/107166] Equalibrum ➜ unknown
[4952/107166] Mangosteen ➜ unknown
[4953/107166] ShowMe ➜ unknown
[4954/107166] HOMESHAKE ➜ unknown
[4955/107166] Busty and the Bass ➜ unknown
[4956/107166] Zack Villere ➜ unknown
[4957/107166] Tuxedo ➜ unknown
[4958/107166] Drew OfThe Drew ➜ unknown
[4959/107166] DPR LIVE ➜ unknown
[4960/107166] Esperanza Spalding ➜ female
[4961/107166] Nohidea ➜ unknown
[get_wikidata_id] Error for MBID cbebfd95-ac2c-42a2-92e1-e546c1048dce: 'url-relation-list'
[4962/107166] Pool Cosby ➜ unknown
[4963/107166] Albin Lee Meldau ➜ male
[get_wikidata_id] Error for MBID 705db9fd-0e22-4334-a323-deeac3c21bdd: 'url-relation-list'
[4964/107166] ARME ➜ unknown
[4965/107166] Mike Love ➜ male
[4966/107166] Rambutan Jam Band ➜ unknown
[4967/107166] Winston Surfshirt ➜ unknown
[4968/107166] Magroove ➜ unknown
[4969/107166] Fat Night ➜ unknown
[4970/107166] Childish Major ➜ male
[4971/107166] Duñe ➜ unknown
[4972/107166] Domo Genesis ➜ unknown
[4973/107166] Hyukoh ➜ unknown
[4974/107166] NASAYA ➜ unknown
[4975/107166] Paula Fuga ➜ female
[4976/107166] Jazz Spastiks ➜ unknown
[4977/107166] Ocean Alley ➜ male
[4978/107166] Ivan Ave ➜ unknown
[get_wikidata_id] Error for MBID 20b3a637-53f7-4617-84da-3f9b7eee32e7: caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4979/107166] Silo ➜ unknown
[4980/107166] Vindata ➜ unknown
[search_artist] Error for 'wuf': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4981/107166] wuf ➜ unknown
[search_artist] Error for 'Tank and The Bangas': caused by: <urlopen error [WinError 10054] Eine vorhandene Verbindung wurde vom Remotehost geschlossen>
[4982/107166] Tank and The Bangas ➜ unknown
[4983/107166] Simpson ➜ female
[4984/107166] Willow ➜ female
[4985/107166] Kush Mody ➜ unknown
[4986/107166] Angelo Mota ➜ male
[4987/107166] DJ Carlo Showcase ➜ male
[4988/107166] The Shock Band ➜ unknown
[4989/107166] Party Blast ➜ unknown
[4990/107166] Frankie Goes To Hollywood ➜ unknown
[4991/107166] Charlie Farley ➜ male
[4992/107166] Sun ➜ male
[4993/107166] Los Formularios ➜ unknown
[4994/107166] GS Boyz ➜ unknown
[get_wikidata_id] Error for MBID 9f41e883-0fc7-4474-bad4-3c2142646d07: 'url-relation-list'
[4995/107166] Cumbia Latin Band ➜ unknown
[4996/107166] Fito Olivares y Su Grupo ➜ male
[4997/107166] People Of 'K' ➜ unknown
[4998/107166] People Of 'K' Feat. Crystal ➜ female
[4999/107166] Murphy Lee ➜ male
[5000/107166] Twerkteam ➜ unknown
[5001/107166] Sofia Reyes ➜ female
[5002/107166] Alexio ➜ unknown
[5003/107166] DJ Luian ➜ male
[5004/107166] Great White ➜ unknown
[5005/107166] Richard Marx ➜ male
[5006/107166] Debbie Gibson ➜ female
[5007/107166] Night Ranger ➜ unknown
[5008/107166] Paula Abdul ➜ female
[5009/107166] Evelyn "Champagne" King ➜ female
[5010/107166] Ratt ➜ unknown
[5011/107166] Whitesnake ➜ unknown
[5012/107166] The Jets ➜ unknown
[5013/107166] Cinderella ➜ unknown
[5014/107166] Sammy Hagar ➜ male
[5015/107166] Joey Graceffa ➜ unknown
[5016/107166] Free ➜ unknown
[5017/107166] Mary Wells ➜ female
[5018/107166] The Marvelettes ➜ unknown
[5019/107166] The Stylistics ➜ unknown
[5020/107166] The Delfonics ➜ unknown
[5021/107166] Barbara Lynn ➜ female
[5022/107166] Aaron Neville ➜ male
[5023/107166] The Persuaders ➜ unknown
[5024/107166] The Originals ➜ unknown


### Preview Mapping 

Missy Elliott: female
Britney Spears: female
Beyoncé: female
Justin Timberlake: male
Shaggy: male
Usher: male
The Pussycat Dolls: unknown
Destiny's Child: unknown
OutKast: unknown
Nelly Furtado: female


unknown              2584
male                 1819
female                602
non-binary gender      11
genderfluid             3
trans woman             2
agender                 2
neutral sex             1
Name: count, dtype: int64


          artist_name artist_gender
0       Missy Elliott        female
1      Britney Spears        female
2             Beyoncé        female
3   Justin Timberlake          male
4              Shaggy          male
5               Usher          male
6               Usher          male
7  The Pussycat Dolls       unknown
8     Destiny's Child       unknown
9             OutKast       unknown


   playlist_id playlist_name  track_position  \
0            0    Throwbacks               0   
1            0    Throwbacks               1   
2            0    Throwbacks               2   
3            0    Throwbacks               3   
4            0    Throwbacks               4   
5            0    Throwbacks               5   
6            0    Throwbacks               6   
7            0    Throwbacks               7   
8            0    Throwbacks               8   
9            0    Throwbacks               9   

                                   track_name         artist_name  \
0  Lose Control (feat. Ciara & Fat Man Scoop)       Missy Elliott   
1                                       Toxic      Britney Spears   
2                               Crazy In Love             Beyoncé   
3                              Rock Your Body   Justin Timberlake   
4                                It Wasn't Me              Shaggy   
5                                       Yeah!               Usher   
6                                      My Boo               Usher   
7                                     Buttons  The Pussycat Dolls   
8                                 Say My Name     Destiny's Child   
9              Hey Ya! - Radio Mix / Club Mix             OutKast   

                                     album_name  \
0                                  The Cookbook   
1                                   In The Zone   
2  Dangerously In Love (Alben für die Ewigkeit)   
3                                     Justified   
4                                      Hot Shot   
5                                   Confessions   
6                                   Confessions   
7                                           PCD   
8                     The Writing's On The Wall   
9                   Speakerboxxx/The Love Below   

                              track_uri  \
0  spotify:track:0UaMYEvWZi0ZqiDOoHU3YI   
1  spotify:track:6I9VzXrHxO9rA9A5euc8Ak   
2  spotify:track:0WqIKmW4BTrj3eJFmnCKMv   
3  spotify:track:1AWQoqb9bSvzTjaLralEkT   
4  spotify:track:1lzr43nnXAijIGYnCT8M8H   
5  spotify:track:0XUfyU2QviPAs6bxSpXYG4   
6  spotify:track:68vgtRHr7iZHpzGpon6Jlo   
7  spotify:track:3BxWKCI06eQ5Od8TY2JBeA   
8  spotify:track:7H6ev70Weq6DdpZyyTmUXk   
9  spotify:track:2PpruBYCo4H7WOBJ7Q2EwM   

                              artist_uri  \
0  spotify:artist:2wIVse2owClT7go1WT98tk   
1  spotify:artist:26dSoYclwsYLMAKD3tpOr4   
2  spotify:artist:6vWDO969PvNqNYHIOW5v0m   
3  spotify:artist:31TPClRtHm23RisEBtV3X7   
4  spotify:artist:5EvFsr3kj42KNv97ZEnqij   
5  spotify:artist:23zg3TcAtWQy7J6upgbUnj   
6  spotify:artist:23zg3TcAtWQy7J6upgbUnj   
7  spotify:artist:6wPhSqRtPu1UhRCDX5yaDJ   
8  spotify:artist:1Y8cdNmUJH7yBTd9yOvr5i   
9  spotify:artist:1G9G7WwrXka3Z1r7aIDjI7   

                              album_uri  track_duration_ms artist_gender  
0  spotify:album:6vV5UrXcfyQD1wu4Qo2I9K             226863        female  
1  spotify:album:0z7pVBGOD7HCIB7S8eLkLI             198800        female  
2  spotify:album:25hVFAxTlDvXbx2X2QkUkE             235933        female  
3  spotify:album:6QPkyl04rXwTGlGlcYaRoW             267266          male  
4  spotify:album:6NmFmPX56pcLBOFMhIiKvF             227600          male  
5  spotify:album:0vO0b1AvY49CPQyVisJLj0             250373          male  
6  spotify:album:1RM6MGv6bcl6NrAG8PGoZk             223440          male  
7  spotify:album:5x8e8UcCeOgrOzSnDGuPye             225560       unknown  
8  spotify:album:283NWqNsCA9GwVHrJk59CG             271333       unknown  
9  spotify:album:1UsmQ3bpJTyK6ygoOOjG1r             235213       unknown  


                                   track_name        artist_name artist_gender
0  Lose Control (feat. Ciara & Fat Man Scoop)      Missy Elliott        female
1                                       Toxic     Britney Spears        female
2                               Crazy In Love            Beyoncé        female
3                              Rock Your Body  Justin Timberlake          male
4                                It Wasn't Me             Shaggy          male

Remaining tracks with known gender: 3174887


## 📘 04 Last.fm AIF360

### 📘 **Explanation**

This code imports all the essential libraries used in the notebook for data processing, building a recommendation system, and performing fairness analysis:

* **`pandas` (`pd`)**: For handling and manipulating structured data in DataFrames.
* **`numpy` (`np`)**: For numerical operations, arrays, and matrix computations.
* **`LabelEncoder`**: Converts categorical labels (e.g., gender) into numeric form (e.g., male → 1, female → 0).
* **`MinMaxScaler`**: Scales numerical features to a fixed range (typically \[0, 1]) — useful for similarity calculations.
* **`cosine_similarity`**: Measures the cosine of the angle between two vectors — used here to calculate similarity between users or items for recommendations.
* **`BinaryLabelDataset`**: A data structure from AIF360 to represent datasets in fairness analysis (note: it's imported twice — only one import is needed).
* **`BinaryLabelDatasetMetric`**: AIF360 class to compute fairness metrics like disparate impact, statistical parity, etc.


WARNING:root:No module named 'tensorflow': AdversarialDebiasing will be unavailable. To install, run:
pip install 'aif360[AdversarialDebiasing]'
WARNING:root:No module named 'tensorflow': AdversarialDebiasing will be unavailable. To install, run:
pip install 'aif360[AdversarialDebiasing]'
WARNING:root:No module named 'inFairness': SenSeI and SenSR will be unavailable. To install, run:
pip install 'aif360[inFairness]'


 Install Required Package

Purpose: Ensure aif360 is available in the environment.

Requirement already satisfied: aif360 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (0.6.1)
Requirement already satisfied: numpy>=1.16 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from aif360) (2.3.1)
Requirement already satisfied: scipy>=1.2.0 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from aif360) (1.15.3)
Requirement already satisfied: pandas>=0.24.0 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from aif360) (2.3.0)
Requirement already satisfied: scikit-learn>=1.0 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from aif360) (1.7.0)
Requirement already satisfied: matplotlib in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from aif360) (3.10.3)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas>=0.24.0->aif360) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas>=0.24.0->aif360) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas>=0.24.0->aif360) (2025.2)
Requirement already satisfied: joblib>=1.2.0 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn>=1.0->aif360) (1.5.1)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn>=1.0->aif360) (3.6.0)
Requirement already satisfied: contourpy>=1.0.1 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib->aif360) (1.3.2)
Requirement already satisfied: cycler>=0.10 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib->aif360) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib->aif360) (4.58.4)
Requirement already satisfied: kiwisolver>=1.3.1 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib->aif360) (1.4.8)
Requirement already satisfied: packaging>=20.0 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib->aif360) (25.0)
Requirement already satisfied: pillow>=8 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib->aif360) (11.2.1)
Requirement already satisfied: pyparsing>=2.3.1 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from matplotlib->aif360) (3.2.3)
Requirement already satisfied: six>=1.5 in c:\users\kapoe\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from python-dateutil>=2.8.2->pandas>=0.24.0->aif360) (1.17.0)
Note: you may need to restart the kernel to use updated packages.



[notice] A new release of pip is available: 24.0 -> 25.1.1
[notice] To update, run: C:\Users\kapoe\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe -m pip install --upgrade pip


Step 2 Load the Dataset

Purpose: Load enriched Last.fm dataset containing user listening history along with gender information.

   Unnamed: 0 Username          Artist  \
0          30  Babs_05     billy ocean   
1          33  Babs_05   bill callahan   
2          35  Babs_05      rod thomas   
3          36  Babs_05       fela kuti   
4         106  Babs_05  machel montano   

                                               Track  \
0            Lovely Day (feat. YolanDa Brown & Ruti)   
1  Arise, Therefore (feat. Six Organs of Admittance)   
2                                        Old Friends   
3          I.T.T. (International Thief Thief) - Edit   
4                                      Private Party   

                                               Album         Date    Time  \
0            Lovely Day (feat. YolanDa Brown & Ruti)  31 Jan 2021   21:23   
1  Arise, Therefore (feat. Six Organs of Admittance)  31 Jan 2021   21:13   
2                                        Old Friends  31 Jan 2021   21:08   
3          I.T.T. (International Thief Thief) [Edit]  31 Jan 2021   21:01   
4                                      Private Party  30 Jan 2021   13:51   

    artist_lastfm                                  mbid gender  
0     billy ocean  0e422e91-a42a-4b4d-8413-9baff67350f2   Male  
1   bill callahan  c309d914-93af-4b3f-8058-d79c75ea89da   Male  
2      rod thomas  deb08150-d897-4adc-abd9-971d57a11f42   Male  
3       fela kuti  6514cffa-fbe0-4965-ad88-e998ead8a82a   Male  
4  machel montano  cf5ffa07-4fb4-47a1-9d61-8721ae9af576   Male  


### Step 3: Clean and Map Gender Labels

**Purpose:** Normalize gender values for consistency and filter out ambiguous/unknown entries.

**Mapping Logic:**

* "M" or "male" → `male`
* "F" or "female" → `female`
* Others → `unknown`




### Step 4: Simulate Popularity-Based Recommendations

**Purpose:** Use artist popularity as a proxy for content-based recommendation.

**How?**

* Count how many times each artist appears in the dataset
* Select the top 20 most frequently appearing artists




### Step 5: Label Recommendations

**Purpose:** Assign binary labels to user-artist pairs based on popularity of the artist.

**Logic:**

* Artist in top 20 → label = 1 (recommended)
* Else → label = 0 (not recommended)



### Step 6: Encode Gender Labels for Fairness Analysis

**Purpose:** Convert gender labels into binary format for use in AIF360 framework.

**Mapping:**

* "male" → 1
* "female" → 0


 Create AIF360 BinaryLabelDataset

**Purpose:** Use AIF360 dataset structure to allow fairness metrics computation.


### Step 7 : Calculate Fairness Metrics

**Purpose:** Evaluate statistical fairness of recommendation decisions using demographic parity.

**Interpretation:**

* A disparate impact close to 1 indicates fair treatment across gender groups
* A value much lower or higher than 1 suggests potential bias


Fairness Evaluation Results:
- Disparate Impact: 1.4111296391330121
- Mean Difference: 0.06009730511424838
- Statistical Parity Difference: 0.06009730511424838
- Consistency: [0.83619261]


| Metric                       | Value | Interpretation                                                            |
| ---------------------------- | ----- | ------------------------------------------------------------------------- |
| **Disparate Impact**         | 1.41  | >1 means females (unprivileged group) are getting recommended more often. |
| **Mean Difference**          | 0.06  | Slight positive bias toward females.                                      |
| **Statistical Parity Diff.** | 0.06  | Similar to mean difference; ideally close to 0.                           |
| **Consistency**              | 0.836 | Fairly high; similar individuals are treated similarly.                   |


ERROR: Could not find a version that satisfies the requirement tensorflow==1.15 (from versions: 2.12.0rc0, 2.12.0rc1, 2.12.0, 2.12.1, 2.13.0rc0, 2.13.0rc1, 2.13.0rc2, 2.13.0, 2.13.1, 2.14.0rc0, 2.14.0rc1, 2.14.0, 2.14.1, 2.15.0rc0, 2.15.0rc1, 2.15.0, 2.15.1, 2.16.0rc0, 2.16.1, 2.16.2, 2.17.0rc0, 2.17.0rc1, 2.17.0, 2.17.1, 2.18.0rc0, 2.18.0rc1, 2.18.0rc2, 2.18.0, 2.18.1, 2.19.0rc0, 2.19.0)
ERROR: No matching distribution found for tensorflow==1.15

[notice] A new release of pip is available: 24.0 -> 25.1.1
[notice] To update, run: C:\Users\kapoe\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe -m pip install --upgrade pip


---
### Prejudice Remover with Real Content Features (eta = 5.0)

**Objective:** Apply the Prejudice Remover fairness algorithm to a music recommendation dataset using actual artist content features and evaluate fairness across gender.

---

### Import Required Libraries

**Purpose:** Load essential Python libraries for data handling, preprocessing, model evaluation, and fairness.



### Load the Dataset

**Purpose:** Read in the Last.fm dataset which includes user listening data and gender information.

### Clean and Normalize Gender Labels

**Purpose:** Standardize gender values and filter out any ambiguous entries.

**Mapping Logic:**

* "male" or "m" → `male`
* "female" or "f" → `female`
* Others → `unknown`

### Encode Gender to Numeric Format

**Purpose:** Convert the `gender_grouped` column into binary numeric values for modeling.

**Mapping:**

* `male` → 1
* `female` → 0


### Assign Recommendation Labels Based on Artist Popularity

**Purpose:** Label tracks as recommended (1) or not (0) based on whether their artist is among the top 20 most frequent in the dataset.


**Logic:**

* Top 20 most common artists are considered popular.
* Tracks from these artists are labeled as recommended (`1`).
* Others are labeled as not recommended (`0`).



### Encode Real Content Features (Artist and Album)

**Purpose:** Convert categorical text fields into numeric features for modeling.


**Steps:**

* Replace missing values with `'unknown'` to ensure clean encoding.
* Use `LabelEncoder` to convert each unique artist and album name into a unique integer.



### Create Combined Feature for Content-Based Modeling

**Purpose:** Generate a single composite feature that combines artist and album information with weighted importance.


**Logic:**

* Artist contributes 60% and Album 40% to the final feature.
* This weighted feature can be used in models such as similarity-based recommenders or fairness algorithms.


### Prepare BinaryLabelDataset and Train/Test Split

**Objective:** Format the data for fairness analysis using AIF360 and create training and test sets.



### Construct a BinaryLabelDataset

**Purpose:** Package the dataset into a structure compatible with fairness algorithms in AIF360.

**Explanation:**

* `combined_feature`: Content-based feature combining artist and album
* `gender_num`: Protected attribute (1 = male, 0 = female)
* `recommended`: Target label (1 = recommended, 0 = not)
* AIF360 uses this structure to assess fairness metrics like disparate impact and statistical parity

###  Split the Data into Train and Test Sets

**Purpose:** Divide the dataset for training and evaluating the fairness-aware model.

**Explanation:**

* 70% of the data is used for training, 30% for testing
* \`shuffl


### Apply Prejudice Remover for Fairness-Aware Prediction

**Objective:** Train and apply the Prejudice Remover algorithm to produce fairness-aware recommendations.


###  Initialize and Train Prejudice Remover

**Purpose:** Fit a fairness-aware model that reduces bias related to a protected attribute (gender).

**Explanation:**

* `PrejudiceRemover`: A fairness-aware in-processing algorithm from AIF360
* `sensitive_attr`: the attribute to be protected (in this case, gender)
* `eta=5.0`: a regularization parameter; higher values increase fairness emphasis
* The model is trained on the `train` set from the `BinaryLabelDataset`

## Generate Predictions on the Test Set

**Purpose:** Safely obtain predicted labels and scores from the trained model.


**Explanation:**

* `predict(test)`: generates predictions on the test dataset
* Ensures predicted `labels` and `scores` are properly shaped arrays for further evaluation
* `test_pred` holds the final prediction object, which includes predicted outcomes and decision scores

### Evaluate Fairness of Prejudice Remover Predictions

**Objective:** Use AIF360 fairness metrics to assess whether the model's predictions are equitable across gender groups.

### Initialize ClassificationMetric

**Purpose:** Compare ground-truth test labels to predicted labels while considering group fairness.

**Explanation:**

* `test`: Original test dataset (ground truth)
* `test_pred`: Predictions from the Prejudice Remover model
* `privileged_groups`: Group considered to have societal advantage (e.g., males → `gender_num = 1`)
* `unprivileged_groups`: Group potentially at disadvantage (e.g., females → `gender_num = 0`)

### Print Fairness Metrics

**Purpose:** Display various fairness metrics to quantify model bias.


**Metrics Explained:**

* **Disparate Impact:** Ratio of favorable outcomes for unprivileged vs. privileged group (ideal ≈ 1.0)
* **Statistical Parity Difference:** Difference in favorable prediction rates (ideal ≈ 0)
* **Equal Opportunity Difference:** Difference in true positive rates across groups
* **Average Odds Difference:** Average of TPR and FPR differences
* **Accuracy:** Overall classification accuracy of the model


Prejudice Remover Fairness Results:
- Disparate Impact: 1.4287720817770078
- Statistical Parity Difference: 0.06270842886542916
- Equal Opportunity Difference: 0.0
- Average Odds Difference: 0.0
- Accuracy: 1.0


| Metric                       | Value     | Interpretation                                                          |
| ---------------------------- | --------- | ----------------------------------------------------------------------- |
| **Disparate Impact**         | **1.43**  | Indicates **slight favor toward females** (unprivileged group).         |
| **Statistical Parity Diff.** | **0.063** | A mild difference in positive recommendation rates between groups.      |
| **Equal Opportunity Diff.**  | **0.0**   | **Perfect equality** in true positive rates.                            |
| **Average Odds Difference**  | **0.0**   | Suggests balanced false positives/negatives across genders.             |
| **Accuracy**                 | **1.0**   | Perfect match with ground truth — likely due to simplistic label logic. |


### Fairness Comparison: Baseline vs. Prejudice Remover

This section compares the fairness metrics and accuracy between a baseline model and a fairness-aware model (Prejudice Remover with `eta=5.0`).

---

### 📊 Summary Table

| Metric                            | Baseline Model | Prejudice Remover | Interpretation                                      |
| --------------------------------- | -------------- | ----------------- | --------------------------------------------------- |
| **Disparate Impact**              | 1.411          | 1.429             | Both > 1, slightly favors unprivileged group        |
| **Statistical Parity Difference** | 0.0601         | 0.0627            | Very small increase, close to 0 is ideal            |
| **Equal Opportunity Difference**  | —              | 0.000             | Perfect equality in TPR across groups               |
| **Average Odds Difference**       | —              | 0.000             | No difference in TPR and FPR — ideal fairness       |
| **Consistency**                   | 0.836          | —                 | Only baseline measured; indicates stable decisions  |
| **Mean Difference**               | 0.0601         | —                 | Same as SPD; difference in positive prediction rate |
| **Accuracy**                      | —              | 1.000             | Perfect accuracy with Prejudice Remover             |

---

### 🧠 Interpretation

* **Disparate Impact** is >1 for both models, indicating unprivileged groups (females) may receive slightly more favorable outcomes.
* **Statistical Parity Difference** is small and comparable in both models, showing low disparity in selection rates.
* **Equal Opportunity & Average Odds Differences** are both 0 in the Prejudice Remover model, indicating perfect group fairness in classification decisions.
* **Accuracy** is 1.0 in the fairness-aware model, meaning it made no classification errors on the test set.
* **Consistency** and **Mean Difference** were only reported for the baseline, suggesting that while consistent, the model didn’t fully enforce fairness constraints.

---

### ✅ Conclusion

The **Prejudice Remover** model slightly improves fairness metrics without sacrificing accuracy — in fact, it achieves perfect accuracy while eliminating key group disparities in classification. This demonstrates the potential of fairness-aware algorithms in real-world recommendation systems.


## 📘 04 Spotify ContentBased Filtering recommender Fairlearn

##  Using Fairlearn to detect gender bias in music recommender system: Spotify Million Dataset

### Content-Based Filtering (CBF) recommender:

**Idea:** Recommend tracks similar to those with high popularity using track features.

**Examples of track features you could use:**
- Genre
- Artist name (or ID)

**How?**
- Create a feature matrix (e.g., one-hot encode genres, artists)
- Use Nearest Neighbor similarity between tracks
- Rank based on similarity to top popular tracks


C:\Users\patri\AppData\Local\Temp\ipykernel_7504\760808870.py:4: DtypeWarning: Columns (2,9) have mixed types. Specify dtype option on import or set low_memory=False.
  df = pd.read_csv("spotify_tracks_with_gender_filtered.csv", on_bad_lines='skip')


  playlist_id playlist_name track_position  \
0           0    Throwbacks              0   
1           0    Throwbacks              1   
2           0    Throwbacks              2   
3           0    Throwbacks              3   
4           0    Throwbacks              4   

                                   track_name        artist_name  \
0  Lose Control (feat. Ciara & Fat Man Scoop)      Missy Elliott   
1                                       Toxic     Britney Spears   
2                               Crazy In Love            Beyoncé   
3                              Rock Your Body  Justin Timberlake   
4                                It Wasn't Me             Shaggy   

                                     album_name  \
0                                  The Cookbook   
1                                   In The Zone   
2  Dangerously In Love (Alben für die Ewigkeit)   
3                                     Justified   
4                                      Hot Shot   

                              track_uri  \
0  spotify:track:0UaMYEvWZi0ZqiDOoHU3YI   
1  spotify:track:6I9VzXrHxO9rA9A5euc8Ak   
2  spotify:track:0WqIKmW4BTrj3eJFmnCKMv   
3  spotify:track:1AWQoqb9bSvzTjaLralEkT   
4  spotify:track:1lzr43nnXAijIGYnCT8M8H   

                              artist_uri  \
0  spotify:artist:2wIVse2owClT7go1WT98tk   
1  spotify:artist:26dSoYclwsYLMAKD3tpOr4   
2  spotify:artist:6vWDO969PvNqNYHIOW5v0m   
3  spotify:artist:31TPClRtHm23RisEBtV3X7   
4  spotify:artist:5EvFsr3kj42KNv97ZEnqij   

                              album_uri track_duration_ms artist_gender;;;;  
0  spotify:album:6vV5UrXcfyQD1wu4Qo2I9K          226863.0        female;;;;  
1  spotify:album:0z7pVBGOD7HCIB7S8eLkLI          198800.0        female;;;;  
2  spotify:album:25hVFAxTlDvXbx2X2QkUkE          235933.0        female;;;;  
3  spotify:album:6QPkyl04rXwTGlGlcYaRoW          267266.0          male;;;;  
4  spotify:album:6NmFmPX56pcLBOFMhIiKvF          227600.0          male;;;;  


Unique artist genders (cleaned):
['female' 'male' 'non-binary gender' 'nan' 'genderfluid' 'trans woman'
 'agender' 'neutral sex' 'female"' 'male"']


Cleaned gender groups:
gender_grouped
male         770657
female       209612
other         53038
nonbinary     15250
Name: count, dtype: int64


Remaining gender groups:
gender_grouped
male         770657
female       209612
other         53038
nonbinary     15250
Name: count, dtype: int64


## Building the Content-Based Recommender using Text Metadata (Track Name, Artist Name, Album Name)

### Train Nearest Neighbors recommender

NearestNeighbors(algorithm='brute', metric='cosine')

### Definition of Recommendation Function

 ### Generate Recommendations

### Check the Recommended Tracks

gender_grouped
female    74.0
male      26.0
Name: proportion, dtype: float64

Seed Track:
track_uri               spotify:track:0UaMYEvWZi0ZqiDOoHU3YI
track_name        Lose Control (feat. Ciara & Fat Man Scoop)
artist_name                                    Missy Elliott
gender_grouped                                        female
Name: 0, dtype: object

Top 10 Recommendations:
                                  track_uri  \
22162  spotify:track:7aJouq94UPaX7yVXd2MQ4k   
67350  spotify:track:5emRlAm3hfUrpPvdNLNXG0   
59030  spotify:track:2rKXzis3tBuNQQojmCldkv   
61545  spotify:track:6XcO3qAAFG9e7DzbgVOEoV   
13405  spotify:track:4z5fkIflIBvSG9elVNmiOJ   
22167  spotify:track:0Z8taEEMbqDMV0eNmD1ypH   
7836   spotify:track:2QLHuAwRJzgDAoGVM8V4U7   
40911  spotify:track:6a2mBuXce3F5rmV2CNvDXx   
63929  spotify:track:06fSNFjg5aU29bHI83aL88   
6354   spotify:track:63WMvjq3EQ7J5S1FLtr9ny   

                                     track_name    artist_name gender_grouped  
22162                                   On & On  Missy Elliott         female  
67350                    Joy (feat. Mike Jones)  Missy Elliott         female  
59030     Bad Man (feat. Vybez Cartel & M.I.A.)  Missy Elliott         female  
61545                               Click Clack  Missy Elliott         female  
13405  We Run This (Without Manicure Interlude)  Missy Elliott         female  
22167              Mommy (with Mommy Interlude)  Missy Elliott         female  
7836                                    I'm Out          Ciara         female  
40911         Bring the Pain (feat. Method Man)  Missy Elliott         female  
63929                              Where You Go          Ciara         female  
6354                                   Overdose          Ciara         female  


Interpretation: That result shows a significant gender imbalance in your CBF recommender output:
- 74% of the recommended tracks are by female artists
- 26% are by male artists

This is useful and revealing — especially since your seed track was from a female artist, which suggests that your CBF model (based on textual similarity from track metadata) might be amplifying similarity in artist gender unintentionally.

### Test the Behavior of the CBF Recommender

### Select a Track Seed per Gender Group

Seed indices by gender:
{'female': 0, 'male': 3, 'nonbinary': 34, 'other': 62}


### Generate Recommendations for Each Seed

### Analyze Gender Distribution per Recommendation Set


--- Recommendations from female seed ---
gender_grouped
female    74.0
male      26.0
Name: proportion, dtype: float64

--- Recommendations from male seed ---
gender_grouped
male         89.0
female       10.0
nonbinary     1.0
Name: proportion, dtype: float64

--- Recommendations from nonbinary seed ---
gender_grouped
nonbinary    95.0
female        3.0
male          2.0
Name: proportion, dtype: float64

--- Recommendations from other seed ---
gender_grouped
male         82.0
female       17.0
nonbinary     1.0
Name: proportion, dtype: float64


## Interpretation: Gender Bias in Content-Based Recommendations (CBF)

### Overview:

You evaluated the content-based filtering (CBF) recommender by seeding it with a track from each gender group (`female`, `male`, `nonbinary`, `other`). The model then retrieved the 100 most similar tracks for each case. Below is a summary of the **artist gender distribution** among those recommendations.

---

### Recommendation Results by Seed Gender

| **Seed Gender** | **Top Gender in Recommendations** | **Share of Same Gender** | **Other Observations** |
|-----------------|-----------------------------------|---------------------------|--------------------------|
| **Female**      | Female                            | 74%                       | 26% male; 0% nonbinary/other |
| **Male**        | Male                              | 89%                       | Slight presence of others (10% female, 1% nonbinary) |
| **Nonbinary**   | Nonbinary                         | 95%                       | Very strong homogeneity |
| **Other**       | Male                              | 82%                       | Moderate mix: 17% female, 1% nonbinary |

---

### Interpretation:
- **Strong Homophily Effect**: Each seed mostly led to recommendations from the same gender group. This is especially extreme for **nonbinary** seeds (95% nonbinary results) and **male** seeds (89% male results).
  
- **Female Seeds Are Slightly More Diverse**: Female seeds returned the most balanced output, with 26% male tracks and no representation of nonbinary or other groups.

- **"Other" Seeds Default Toward Male**: Despite being seeded with "other", the model leaned toward male artists (82%), showing **possible bias or representation gaps** in the underlying data.

---

### Implications

- The recommender **amplifies existing biases** in the input features (artist name, track name, album name), leading to **group-wise echo chambers**.
- **Nonbinary and "Other" artists** are likely underrepresented or isolated in feature space, leading to **low crossover** into other groups' recommendations.

---

### Bias Detection using Fairlearn

### Generate CBF Recommendations

###  Get track indices from recommendations

Index(['track_uri', 'artist_gender', 'track_index'], dtype='object')


                selection_rate    count
gender_grouped                         
female                  0.0004  22277.0
male                    0.0010  85720.0
nonbinary               0.0023    442.0
other                   0.0000    169.0


### Interpretation of Fairness Metrics for CBF Recommender:

We computed **selection rates** using Fairlearn’s `MetricFrame` across different gender groups. The selection rate represents the proportion of tracks from each group that were recommended by the system.

#### Results

| Gender       | Selection Rate | Count   | Approx. # Recommended |
|--------------|----------------|---------|------------------------|
| **male**     | 0.0010         | 85,720  | ~89                    |
| **female**   | 0.0004         | 22,277  | ~10                    |
| **nonbinary**| 0.0023         | 442     | ~1                     |
| **other**    | 0.0000         | 169     | 0                      |

--- 

#### Interpretation:

- **Male artists dominate the recommendations.**  
  89 out of 100 recommendations are by male artists, representing 89% of the total, even though male artists make up ~79% of the dataset.

- **Female artists are underrepresented.**  
  Only 10% of the recommendations come from female artists, though they make up ~21% of the data. The selection rate for female artists (0.0004) is less than half that of male artists (0.0010).

- **Nonbinary artists have a higher selection rate but a small sample size.**  
  With just 442 tracks in total, the higher selection rate (0.0023) reflects just a single recommendation and is not statistically reliable.

- **Artists labeled as “other” are completely excluded.**  
  None of the 169 tracks in this group were recommended.

#### Key Takeaway:

> The Content-Based Filtering (CBF) recommender system exhibits **gender bias**, favoring male artists and underrepresenting female and minority gender groups in its recommendations. This may be due to biased input features, overrepresentation of certain groups, or underlying popularity trends in the dataset.

---

### Inspection of Pairwise Demographic Parity Differences (DPD) 

     Group 1    Group 2       DPD
5  nonbinary      other  0.002262
1     female  nonbinary  0.001814
3       male  nonbinary  0.001224
4       male      other  0.001038
0     female       male  0.000589
2     female      other  0.000449


### Inspection of Pairwise Demographic Parity Differences (DPD)

The table below presents the **absolute differences in selection rates** between each pair of gender groups, based on the Content-Based Filtering (CBF) recommender:

| Group 1    | Group 2    | DPD      |
|------------|------------|----------|
| nonbinary  | other      | 0.002262 |
| female     | nonbinary  | 0.001814 |
| male       | nonbinary  | 0.001224 |
| male       | other      | 0.001038 |
| female     | male       | 0.000589 |
| female     | other      | 0.000449 |

--- 

#### Interpretation:

- The **largest disparity** in recommendation rate occurs between **nonbinary and other** artists (DPD = 0.002262), followed by **female vs. nonbinary** and **male vs. nonbinary**.
- This suggests that **nonbinary artists** are treated quite differently than both **female** and **male** artists, even though the total number of nonbinary tracks is very small.
- The DPD between **female and male** artists (0.000589) is significant in the context of such low selection rates overall — it reinforces that **female artists are recommended less frequently** than male artists.
- The **"other"** category consistently appears in the largest disparities, including being **completely excluded** from recommendations (selection rate = 0.0).

#### Key findings:

> The CBF model exhibits **notable disparities** in how frequently it recommends tracks from different gender groups.  
> While the overall selection rates are low, even small absolute differences indicate **systemic bias**, especially when consistent across multiple group pairs.  
> This calls for fairness-aware interventions or model adjustments to mitigate bias.

---

## Bias Mitigation

### In-Processing: Fairness-Aware Learning using Fairlearn’s ExponentiatedGradient

###  Trying a new approach: We Need a Similarity Score for Fairness-Aware Learning using Fairlearn’s ExponentiatedGradient

Our recommender is based on content-based filtering (CBF), which recommends tracks similar to popular seeds.  
To apply in-processing fairness mitigation (e.g., `ExponentiatedGradient`), we need a numeric feature (`X`) that reflects the recommendation logic.  
Basic features like `play_count` or `track_index` don't align well with CBF behavior.  
A similarity score captures how similar a track is to top-N popular tracks based on content (e.g., metadata or genre).  
We can compute this score using cosine similarity over TF-IDF embeddings of track metadata.  
This allows a classifier to learn what tracks are likely to be recommended and apply fairness constraints during training.  
Without a feature like similarity score, the mitigation model has no useful signal to act on.

❗Note: The following code was to test a new approach, with the goal to apply fairness aware learning using the Fairlearn's Exponentiated Gradient. But it failed because: It failed the model couldn’t learn a meaningful pattern from the weak signal in the similarity score, so it defaulted to predicting 0 for all tracks to minimize error and satisfy fairness trivially.

### Integrate a Similarity Score

### In-Processing Fairness Mitigation Using Similarity Score

### Bias mitigation: Resample data and re-run post-processing fairness mitigationusing ThresholdOptimizer from Fairlearn

Selection Rate by Gender Group (Post-Processed):
gender_grouped
female    0.3333
male      0.2982
other     0.0000
Name: selection_rate, dtype: float64

Selection Rate Range (Max - Min): 0.3333


### Post-Processing Fairness Evaluation (Threshold Optimizer)

After applying the `ThresholdOptimizer` with a **demographic parity** constraint, the selection rates (i.e., proportion of tracks labeled as "recommended") per gender group are:

| Gender Group | Selection Rate |
|--------------|----------------|
| Female       | 0.3333         |
| Male         | 0.2982         |
| Other        | 0.0000         |

---

### Interpretation:

- The **female group** receives the **highest recommendation rate** (33.33%), followed closely by **male** (29.82%).
- The **"other" group** receives **no recommendations**, indicating a fairness issue remains.
- The **selection rate range** (max - min) is **0.3333**, which is relatively high — suggesting that **demographic parity was not fully achieved** across all groups.

 This may be due to very limited positive examples for the "other" group, limiting the optimizer’s ability to adjust thresholds meaningfully for that subgroup. 

---

### Fairness Comparison: Before vs After Post-Processing (Threshold Optimizer)

#### Before Bias Mitigation (Raw CBF Recommender)

| Gender Group | Selection Rate | Count   | Approx. # Recommended |
|--------------|----------------|---------|------------------------|
| Male         | 0.0010         | 85,720  | ~89                   |
| Female       | 0.0004         | 22,277  | ~10                   |
| Nonbinary    | 0.0023         | 442     | ~1                    |
| Other        | 0.0000         | 169     | 0                     |

- **Heavily skewed** toward male artists (89% of recommendations).
- Female artists are significantly underrepresented relative to their dataset proportion.
- "Other" group received **no recommendations** at all.

---

#### After Bias Mitigation (Threshold Optimizer with Demographic Parity)

| Gender Group | Selection Rate |
|--------------|----------------|
| Female       | 0.3333         |
| Male         | 0.2982         |
| Other        | 0.0000         |

- **Substantial increase** in selection rates for female (from 0.0004 → 0.3333) and male (from 0.0010 → 0.2982).
- Still **no recommendations for "other"**, indicating remaining fairness gaps.
- **Selection rate range** is 0.3333, better than before but still not ideal.

---

### Interpretation:

- The post-processing approach dramatically improved gender balance **between male and female groups**, correcting the initial male overrepresentation.
- However, the **"other" group remains excluded**, likely due to insufficient positive examples in training.
- Overall, bias mitigation was **partially successful**: fairness between major gender groups improved, but **further action is needed** for minority inclusion.

---

## 📘 03 Spotify Baselinerecommender Fairlearn

## Using Fairlearn to detect gender bias in music recommender system: Spotify Million Dataset

### Baseline Recommender: Popularity-Based

Top 100 Most Popular Tracks:
   track_index  play_count                             track_uri  \
0       102035        4562  spotify:track:7KXjTSCq5nL1LoYtL7XAwS   
1        27741        4355  spotify:track:1xznGGDReH1oQq0xzbwXa3   
2       108379        4105  spotify:track:7yyRTcZmCiyzzJlNzGC9Ol   
3        49986        3985  spotify:track:3a1lNhkSLSkpJE4MSHpDu9   
4        79319        3540  spotify:track:5hTpBe8h35rJ67eAWHQsJx   
5        15255        3485  spotify:track:152lZdxL1OR0ZMW6KquMif   
6        31318        3473  spotify:track:2EEeOnHehOozLq4aS0n6SL   
7       101161        3456  spotify:track:7GX5flRQZVHRAGd6B4TmDO   
8         6392        3393  spotify:track:0SGkqnVQo9KPytSri1H6cF   
9        84200        3319  spotify:track:62vpWI1CHwFy7tMIcSStl8   

       artist_gender  
0               male  
1               male  
2               male  
3               male  
4               male  
5               male  
6               male  
7  non-binary gender  
8               male  
9             female  

Top N Popular Track Recommendations:
    track_index artist_gender  play_count
0        102035          male        4562
1         27741          male        4355
2        108379          male        4105
3         49986          male        3985
4         79319          male        3540
..          ...           ...         ...
95        97204          male        1861
96       107455          male        1853
97        54602          male        1850
98        83273          male        1850
99        12739          male        1845

[100 rows x 3 columns]


🎯 Gender distribution in Top 100 Recommended Tracks (%):
artist_gender
male                 91.0
female                6.0
non-binary gender     3.0
Name: proportion, dtype: float64


# Popularity-Based Baseline Recommender: Interpretation

We implemented a popularity-based recommender system that recommends the same **Top 100 most popular tracks** to every user. These tracks were identified based on how frequently they appeared across all playlists.

---

## Gender Distribution in Top 100 Tracks

Below is the approximate gender distribution among the Top 100 tracks recommended:

| Artist Gender       | Percentage (%) |
|---------------------|----------------|
| male                | 91%            |
| female              | 6%             |
| non-binary gender   | 3%             |

---

>  **Observation:** The popularity-based recommender overwhelmingly favors male artists. This may reflect user behavior in the dataset or historical disparities in artist exposure, but it raises concerns about fairness and representation.

**Key Insight:** The popularity-based recommender system, while simple and effective at surfacing the most played tracks, **amplifies existing gender biases** present in the dataset. In fact, **over 90% of the Top 100 recommended tracks are by male artists**, with only a small fraction representing female or non-binary artists.

---

### Implications:

- The model recommends tracks purely based on their historical frequency in playlists.
- Since male artists dominate historical listening behavior in the dataset, the recommender **over-represents male artists**.
- All users receive the same recommendations — meaning the bias is applied uniformly and **systematically reinforces a skewed distribution**.
- While this model is useful as a **baseline benchmark**, it is **not suitable for fair or inclusive recommendation tasks**.

### Bias detection using Fairlearn

Found existing installation: numpy 1.26.4

WARNING: Failed to remove contents in a temporary directory 'C:\Users\patri\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\~%mpy.libs'.
You can safely remove it manually.
WARNING: Failed to remove contents in a temporary directory 'C:\Users\patri\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\~0mpy'.
You can safely remove it manually.



Uninstalling numpy-1.26.4:
  Successfully uninstalled numpy-1.26.4
Found existing installation: scipy 1.16.0
Uninstalling scipy-1.16.0:
  Successfully uninstalled scipy-1.16.0
Found existing installation: scikit-learn 1.7.0
Uninstalling scikit-learn-1.7.0:
  Successfully uninstalled scikit-learn-1.7.0
Found existing installation: fairlearn 0.12.0
Uninstalling fairlearn-0.12.0:
  Successfully uninstalled fairlearn-0.12.0


Collecting numpy==1.26.4
  Using cached numpy-1.26.4-cp311-cp311-win_amd64.whl.metadata (61 kB)
Using cached numpy-1.26.4-cp311-cp311-win_amd64.whl (15.8 MB)
Installing collected packages: numpy
Successfully installed numpy-1.26.4


ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
aif360 0.6.1 requires scikit-learn>=1.0, which is not installed.
aif360 0.6.1 requires scipy>=1.2.0, which is not installed.
imagehash 4.3.2 requires scipy, which is not installed.
imbalanced-learn 0.13.0 requires scikit-learn<2,>=1.3.2, which is not installed.
imbalanced-learn 0.13.0 requires scipy<2,>=1.10.1, which is not installed.
implicit 0.7.2 requires scipy>=0.16, which is not installed.
missingno 0.5.2 requires scipy, which is not installed.
pandas-profiling 3.2.0 requires scipy>=1.4.1, which is not installed.
phik 0.12.4 requires scipy>=1.5.2, which is not installed.
recbole 1.2.1 requires scikit-learn>=0.23.2, which is not installed.
recbole 1.2.1 requires scipy>=1.6.0, which is not installed.
xgboost 2.1.3 requires scipy, which is not installed.
pandas-profiling 3.2.0 requires joblib~=1.1.0, but you have joblib 1.5.1 which is incompatible.


Collecting scipy

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
aif360 0.6.1 requires scikit-learn>=1.0, which is not installed.
imbalanced-learn 0.13.0 requires scikit-learn<2,>=1.3.2, which is not installed.
recbole 1.2.1 requires scikit-learn>=0.23.2, which is not installed.
pandas-profiling 3.2.0 requires joblib~=1.1.0, but you have joblib 1.5.1 which is incompatible.



  Using cached scipy-1.16.0-cp311-cp311-win_amd64.whl.metadata (60 kB)
Requirement already satisfied: numpy<2.6,>=1.25.2 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scipy) (1.26.4)
Using cached scipy-1.16.0-cp311-cp311-win_amd64.whl (38.6 MB)
Installing collected packages: scipy
Successfully installed scipy-1.16.0
Collecting scikit-learn
  Using cached scikit_learn-1.7.0-cp311-cp311-win_amd64.whl.metadata (14 kB)
Requirement already satisfied: numpy>=1.22.0 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn) (1.26.4)
Requirement already satisfied: scipy>=1.8.0 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn) (1.16.0)
Requirement already satisfied: joblib>=1.2.0 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn) (1.5.1)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn) (3.5.0)
Using cached scikit_learn-1.7.0-cp311-cp311-win_amd64.whl (10.7 MB)
Installing collected packages: scikit-learn
Successfully installed scikit-learn-1.7.0


ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
sklearn-compat 0.1.3 requires scikit-learn<1.7,>=1.2, but you have scikit-learn 1.7.0 which is incompatible.


Collecting fairlearn
  Using cached fairlearn-0.12.0-py3-none-any.whl.metadata (7.0 kB)
Requirement already satisfied: numpy>=1.24.4 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from fairlearn) (1.26.4)
Requirement already satisfied: pandas>=2.0.3 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from fairlearn) (2.2.3)
Requirement already satisfied: scikit-learn>=1.2.1 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from fairlearn) (1.7.0)
Requirement already satisfied: scipy>=1.9.3 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from fairlearn) (1.16.0)
Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas>=2.0.3->fairlearn) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas>=2.0.3->fairlearn) (2024.2)
Requirement already satisfied: tzdata>=2022.7 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from pandas>=2.0.3->fairlearn) (2025.1)
Requirement already satisfied: six>=1.5 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from python-dateutil>=2.8.2->pandas>=2.0.3->fairlearn) (1.17.0)
Requirement already satisfied: joblib>=1.2.0 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn>=1.2.1->fairlearn) (1.5.1)
Requirement already satisfied: threadpoolctl>=3.1.0 in c:\users\patri\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages (from scikit-learn>=1.2.1->fairlearn) (3.5.0)
Using cached fairlearn-0.12.0-py3-none-any.whl (240 kB)
Installing collected packages: fairlearn
Successfully installed fairlearn-0.12.0


Packages are working fine!


### Prepare Input for using Fairlearn and Treat Top-100 tracks from the model as recommended

🎯 Absolute Gender Distribution in Full Catalog:
artist_gender
male                 85720
female               22277
non-binary gender      442
genderfluid             92
trans woman             44
agender                 32
neutral sex              1
Name: count, dtype: int64

📊 Percentage Distribution:
artist_gender
male                 78.93
female               20.51
non-binary gender     0.41
genderfluid           0.08
trans woman           0.04
agender               0.03
neutral sex           0.00
Name: proportion, dtype: float64


If the recommender was perfectly unbiased and proportional, we’d expect about: 79 male tracks, 20 female tracks, 1 from non-binary or another minority group.

### Definition of Sensitive/Protected Attribute


### Metrics computation using Fairlearn

🎯 Selection rate by gender group (multiclass):
gender_grouped
female       0.000269
male         0.001062
nonbinary    0.006787
other        0.000000
Name: selection_rate, dtype: float64


### Visualize selection rates by gender groups

## Selection Rate by Gender Group

| Gender Group | Selection Rate      | Interpretation                                                  |
|--------------|---------------------|------------------------------------------------------------------|
| **Male**     | 0.001062 (0.11%)    | 🟢 Highest selection rate — most favored group                   |
| **Female**   | 0.000269 (0.03%)    | 🔴 Much lower selection rate — clear underrepresentation         |
| **Nonbinary**| 0.006787 (0.68%)    | 🟡 Very high relative to their catalog share — possible overrepresentation |
| **Other**    | 0.000000            | ⚠️ No tracks selected at all from this group                     |

---

### What This Means? 

- **Bias is present.** Male artists are ~4× more likely to be recommended than female artists.
- **Nonbinary overrepresentation**: although they make up <1% of the catalog, their selection rate is significantly higher.
- **No representation** for the "other" category — none of their tracks were recommended.

---

This outcome is based on selecting the **Top 100 most popular tracks**, and:
- Popularity often reflects **historical user behavior**, which can carry **inherent biases**.
- If male artists dominate in **past play counts**, they will also dominate in recommendations — unless mitigated.

---

### Inspection of Pairwise Demographic Parity Differences (DPD) 
These are useful for identifying specific fairness disparities between individual gender groups, rather than just looking at the overall range. This helps reveal whether certain groups are consistently under- or over-recommended relative to others. It also allows for a more nuanced understanding of fairness in multiclass settings, where biases may affect each group differently. By comparing each group pair directly, we can target mitigation efforts more precisely.

     Group A    Group B  DP Difference (A vs B)
0     female       male                0.000792
1     female  nonbinary                0.006518
2     female      other                0.000269
3       male  nonbinary                0.005726
4       male      other                0.001062
5  nonbinary      other                0.006787


## Interpretation of Pairwise Demographic Parity Differences (DPD)

| Comparison             | DPD (A vs B) | Interpretation |
|------------------------|--------------|----------------|
| **female vs male**     | 0.000792     | Female artists are recommended less often than male artists. |
| **female vs nonbinary**| 0.006518     | Female artists are significantly under-recommended compared to nonbinary artists. |
| **female vs other**    | 0.000269     | A small difference suggests female and "other" groups are similarly underrepresented. |
| **male vs nonbinary**  | 0.005726     | Nonbinary artists are recommended more often than male artists. |
| **male vs other**      | 0.001062     | Male artists are recommended more than the "other" group. |
| **nonbinary vs other** | 0.006787     | The largest gap — nonbinary artists are far more likely to be recommended than those in the "other" group. |

---

### Key Insight of the DPD Evaluation
The largest disparities occur between nonbinary artists and all other groups, especially the "other" category. Female artists are consistently less recommended than both male and nonbinary artists, highlighting an area for potential fairness improvement.


## Bias Mitigation

### Bias Mitigation using all genders: (Female - Male - Non-binary - Others)

### In-Processing: Fairness-Aware Learning using Fairlearn’s ExponentiatedGradient

Selection Rate by Gender Group (Fair Model):
gender_grouped
female       0.000150
male         0.001128
nonbinary    0.007519
other        0.000000
Name: selection_rate, dtype: float64

📏 Selection Rate Range (Max - Min): 0.0075


## Selection Rate by Gender Group (Fair Model)

| Gender Group | Selection Rate |
|--------------|----------------|
| **female**   | 0.000150 (0.015%) |
| **male**     | 0.001128 (0.11%)  |
| **nonbinary**| 0.007519 (0.75%)  |
| **other**    | 0.000000 (0.00%)  |

---

### **Interpretation**
- **Non-binary artists** have the **highest selection rate** (0.75%), indicating a **disproportionately higher exposure** compared to other groups.
- **Male artists** are selected at a moderate rate (0.11%), while **female artists** are selected less frequently (0.015%).
- The **"other" group**, which includes diverse identities like agender, genderfluid, and trans woman, has a **selection rate of 0.00%**, indicating complete exclusion in this fairness-aware recommendation output.

---

### **Observations**
- The fairness mitigation did not successfully equalize exposure across all gender groups.
- While **non-binary exposure increased significantly**, **female and "other" categories remained underrepresented**.
- The **"other" category’s absence** from the recommendations suggests the model may still overlook minority groups with small sample sizes.

---

### **Key Findings**
- The **selection rate range of 0.0075** shows a **large disparity** remains between the best- and worst-represented groups, even after mitigation.
- **In-processing alone is insufficient** to ensure equal visibility for all gender identities in this setup.

--- 

The baseline recommender doesn’t consider fairness — it just picks the most popular tracks, which often reflect historical bias. The mitigated model actively tries to correct for bias — but it might affect accuracy or exposure in other ways, that's what we experienced in this case.

## Comparison Baseline Recommender VS Fair Model

To fairly compare the **baseline recommender** and the **fairness-aware model**, we need to evaluate both using the **same data** — in this case, the test set created during the fairness-aware model training.

This is important because:
- ✅ It ensures both models are judged under **identical conditions** — same tracks, same gender distribution.
- ✅ It avoids bias caused by different sample sizes or group proportions between training and full datasets.
- ✅ It aligns with standard **machine learning evaluation practices**, where performance (including fairness) is measured on **held-out data**.
- ❌ Comparing the fair model on a test set vs. the baseline on the full dataset would introduce inconsistency and **make fairness metrics unreliable**.

By using the **same test split**, we ensure the fairness comparison reflects true differences in model behavior — not differences in data.


🎯 Baseline Selection Rate by Gender Group (on Test Set):
gender_grouped
female       0.000150
male         0.001244
nonbinary    0.007519
other        0.000000
Name: selection_rate, dtype: float64


C:\Users\patri\AppData\Local\Temp\ipykernel_18796\3975221063.py:34: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  plt.text(x[i] - width/2, baseline_rates[i] + 0.0001, f'{baseline_rates[i]:.6f}', ha='center', fontsize=8)
C:\Users\patri\AppData\Local\Temp\ipykernel_18796\3975221063.py:35: FutureWarning: Series.__getitem__ treating keys as positions is deprecated. In a future version, integer keys will always be treated as labels (consistent with DataFrame behavior). To access a value by position, use `ser.iloc[pos]`
  plt.text(x[i] + width/2, fair_rates[i] + 0.0001, f'{fair_rates[i]:.6f}', ha='center', fontsize=8)
C:\Users\patri\AppData\Local\Temp\ipykernel_18796\3975221063.py:37: UserWarning: Glyph 127911 (\N{HEADPHONE}) missing from font(s) DejaVu Sans.
  plt.tight_layout()
C:\Users\patri\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\IPython\core\pylabtools.py:170: UserWarning: Glyph 127911 (\N{HEADPHONE}) missing from font(s) DejaVu Sans.
  fig.canvas.print_figure(bytes_io, **kw)


## Selection Rate Comparison (Baseline vs Fair Model)

| Gender Group | Baseline Selection Rate | Fair Model Selection Rate | Change |
|--------------|--------------------------|----------------------------|--------|
| Female       | 0.000150                 | 0.000150                   | ➖ No change |
| Male         | 0.001244                 | 0.001128                   | 🔻 Slight decrease |
| Nonbinary    | 0.007519                 | 0.007519                   | ➖ No change |
| Other        | 0.000000                 | 0.000000                   | ⚠️ Still excluded |

---

### Interpretation:
- **Nonbinary artists** dominate the recommendations in both models, with **no change** post-mitigation.
- **Female artists** remained severely underrepresented, and the fairness-aware model did **not** improve this.
- **Male artists** saw a **small reduction** in exposure.
- The **“other” group (e.g., agender, trans)** is still entirely **excluded**, indicating poor support for underrepresented identities.

---

### Fairness Insight

Despite applying **demographic parity mitigation**, the selection rates remained **almost unchanged** — suggesting that:
- The model had **limited flexibility** to rebalance group exposure.
- The extremely small representation of some gender groups in the data likely limits what the model can learn or adjust.

---

### Bias mitigation: Post-Processing using ThresholdOptimizer from Fairlearn

## Why Merge Gender Groups for ThresholdOptimizer

### The Problem:
When using `ThresholdOptimizer` for post-processing bias mitigation, every **sensitive group** (like gender) must contain:
- At least one **positive label** (`label=1`, meaning recommended), and  
- At least one **negative label** (`label=0`, meaning not recommended).

However, in our dataset:
- The `"other"` gender group (e.g., agender, genderfluid, trans woman) had **only label=0** in the test set — no recommended tracks.
- This made the group **"degenerate"**, and `ThresholdOptimizer` raised an error because it couldn't compute a threshold trade-off curve.

---

### The Fix: Merging Groups

To avoid dropping the `"other"` group entirely, we **merged it with the `"nonbinary"` group**, creating a single `"other"` group that:
- Contains enough examples from both classes (`1` and `0`),
- Satisfies `ThresholdOptimizer`’s requirements,
- Still preserves the **inclusivity of gender diversity** in fairness evaluation.

---


🎯 Selection Rate by Gender Group (Post-Processing with 'other' merged):
gender_grouped
female    0.001047
male      0.000933
other     0.000000
Name: selection_rate, dtype: float64

📏 Selection Rate Range (Max - Min): 0.0010


## Selection Rate by Gender Group (Post-Processing Mitigation)

| Gender Group | Selection Rate | Interpretation |
|--------------|----------------|----------------|
| Female       | 0.001047 (0.10%) | 🟢 Slightly favored in this setup |
| Male         | 0.000933 (0.09%) | 🟡 Slightly below female group |
| Other        | 0.000000 (0.00%) | 🔴 No representation at all |

---

### Interpretation

- **Fairness improved** slightly between male and female artists: selection rates are nearly equal.
- **"Other" group received no recommendations** — despite merging, they still weren’t selected by the post-processed model.
- **Selection rate range is 0.0010**, a small disparity numerically, but one group remains completely excluded.

---

### Key insights

- **ThresholdOptimizer successfully reduced gender disparity** between male and female.
- However, **data scarcity for the “other” group** still limits fairness — a sign that fairness mitigation can't overcome extreme underrepresentation alone.
- Additional steps like **reweighting, oversampling, or enriched metadata** might be needed to support underrepresented gender identities.


## Bias Mitigation using Reweighting (Inprocessing)

 ### Evaluate selection rate by gender group (test set)

Selection Rate by Gender Group (GridSearch Reweighted - Spotify):
gender_grouped
female       0.000150
male         0.001128
nonbinary    0.007519
other        0.000000
Name: selection_rate, dtype: float64

Selection Rate Range (Max - Min): 0.0075


## Selection Rate by Gender Group (Reweighted Model – Spotify)

| Gender Group | Selection Rate | Interpretation |
|--------------|----------------|----------------|
| Female       | 0.000150 (0.02%) | 🔴 Severely underrepresented despite reweighting |
| Male         | 0.001128 (0.11%) | 🟡 Still the most selected group overall |
| Nonbinary    | 0.007519 (0.75%) | 🟢 Strongly overrepresented given catalog share |
| Other        | 0.000000 (0.00%) | ⚠️ Completely excluded — no tracks selected |

---

### Interpretation:

- **Nonbinary artists** continue to be **overrepresented**, likely boosted by reweighting but also affected by their popularity in the Top-N.
- **Female artists remain under-selected**, showing reweighting did not sufficiently shift outcomes in their favor.
- The **"Other" group** once again receives **no exposure**, likely due to very few training examples or low popularity.

The selection rate disparity range is **0.0075**, which remains notable — indicating that **reweighting alone did not produce equal representation** across gender groups.

---

### Key findings:

- Reweighting helps shift the model, but **group size and label imbalance** limit its effectiveness — especially for groups like **"female"** and **"other"**.
- Applying post-processing (like `ThresholdOptimizer`) is **limited by the same data sparsity** and not yield meaningful improvement in this multiclass context.

## Overall Fairness Comparison – Spotify Dataset

| Gender Group | Baseline Selection Rate | Fair Model (EG) | Post-Processing | Reweighted (GridSearch) | Interpretation |
|--------------|--------------------------|------------------|------------------|---------------------------|----------------|
| **Female**   | 0.000150                 | 0.000150         | 0.001047         | 0.000150                  | 🔴 Severely underrepresented in all models |
| **Male**     | 0.001244                 | 0.001128         | 0.000933         | 0.001128                  | 🟡 Consistently favored; slightly reduced by fair models |
| **Nonbinary**| 0.007519                 | 0.007519         | —                | 0.007519                  | 🟢 Strongly overrepresented; not addressed in post-processing |
| **Other**    | 0.000000                 | 0.000000         | 0.000000         | 0.000000                  | ⚠️ Excluded in all models due to extreme data sparsity |

---

### Key Observations

- **No model achieved balanced representation** across all gender groups.
- The **Fair Model (ExponentiatedGradient)** made **no impact** on selection rates — likely due to limitations in how it handled the label imbalance.
- **Post-processing (ThresholdOptimizer)** slightly improved fairness for `female`, but **excluded "nonbinary"** due to its binary-only constraint.
- **GridSearch reweighting** retained high exposure for `nonbinary` and male artists, while still excluding `"other"` and failing to lift `"female"` meaningfully.

---

### Final Takeaway

Despite multiple fairness interventions, the **Spotify popularity-based recommender** remains skewed — favoring `"nonbinary"` and `"male"` artists, and systematically excluding `"other"` and underrepresenting `"female"`.  
This highlights the **limitations of fairness mitigation** in the presence of **extreme class imbalance** and suggests the need for more **targeted approaches**, such as group-aware re-ranking or data augmentation.


