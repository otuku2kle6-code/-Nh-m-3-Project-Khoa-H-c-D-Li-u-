# Task: Explore & Understand the Spotify Dataset (EDA)

**Assigned to:** [ your teammate's name ]  
**Step in project:** Step 3 — Exploratory Data Analysis (EDA)  
**Estimated time:** 3–5 hours  
**Branch name to work on:** `feature/eda`

---

## What is this task?

Before we build any machine learning model, we need to **understand our data first**.  
This step is called **EDA — Exploratory Data Analysis**.

Think of it like this: before cooking a meal, you open the fridge and check what ingredients you have, whether anything is expired, and what goes well together. EDA is that "opening the fridge" step for data.

By the end of this task, you should be able to answer:
- What does the dataset look like? How many rows and columns?
- Are there any missing or broken values?
- Which features (columns) are most related to `popularity`?
- What does the popularity distribution look like?
- Which genres tend to be most popular?

---

## Part 1 — Setup (do this first)

### Step 1.1 — Clone the repo and create your branch

Open your terminal in VS Code (`Ctrl + ~`) and run:

```bash
git clone https://github.com/YOUR-TEAM/spotify-project.git
cd spotify-project
git checkout -b feature/eda
```

### Step 1.2 — Install the libraries

Run this in the terminal:

```bash
pip install pandas numpy matplotlib seaborn
```

> **What are these?**
> - `pandas` — loads and works with table data (like Excel in Python)
> - `numpy` — math operations on numbers and arrays
> - `matplotlib` — draws charts and graphs
> - `seaborn` — draws prettier statistical charts, built on top of matplotlib

### Step 1.3 — Download the dataset

Download `dataset.csv` from Kaggle:  
https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Place it inside the `data/` folder of the project:
```
spotify-project/
  └── data/
      └── dataset.csv   ← put it here
```

### Step 1.4 — Create your notebook

In VS Code, create a new file:
```
notebooks/eda.ipynb
```
Make sure you have the **Jupyter** extension installed in VS Code.  
Open the file → click "Select Kernel" → choose your Python environment.

---

## Part 2 — Exercises (do these one by one)

Each exercise has an explanation, then the code to run, then a question for you to answer.

---

### Exercise 1 — Load the data and take a first look

**What you're learning:** How to load a CSV file and see its basic structure.

```python
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('../data/dataset.csv')

# How many rows and columns?
print(df.shape)

# Show the first 5 rows
df.head()
```

**Then run:**
```python
# See column names and data types
df.info()
```

**Then run:**
```python
# Basic statistics: mean, min, max for each numeric column
df.describe()
```

> **Answer these questions in a comment in your notebook:**
> 1. How many rows (songs) are in the dataset?
> 2. How many columns (features) are there?
> 3. What data type is the `popularity` column? (int, float, or object?)
> 4. What is the average (mean) popularity score?

---

### Exercise 2 — Check for missing values and duplicates

**What you're learning:** Real datasets are messy. You need to find and handle broken data before modeling.

```python
# Count missing values per column
print("Missing values per column:")
print(df.isnull().sum())
```

```python
# Check for duplicate rows (same song appearing twice)
duplicates = df.duplicated(subset=['track_id']).sum()
print(f"Number of duplicate track IDs: {duplicates}")
```

```python
# How many songs have popularity = 0?
zero_pop = (df['popularity'] == 0).sum()
print(f"Songs with popularity = 0: {zero_pop}")
print(f"That is {zero_pop / len(df) * 100:.1f}% of the dataset")
```

> **Answer these questions:**
> 1. Are there any columns with missing values? Which ones?
> 2. Are there duplicate songs? What does that mean for our analysis?
> 3. Is the number of popularity=0 songs a problem? Why or why not?

---

### Exercise 3 — Explore the popularity column (our target)

**What you're learning:** Understanding the variable we want to predict is the most important thing in any ML project.

```python
import matplotlib.pyplot as plt

# Plot the distribution of popularity
df['popularity'].hist(bins=30, figsize=(8, 4), color='steelblue', edgecolor='white')
plt.title('Distribution of Popularity Scores')
plt.xlabel('Popularity (0-100)')
plt.ylabel('Number of songs')
plt.show()
```

```python
# Now filter out the 0-popularity songs and plot again
df_filtered = df[df['popularity'] > 0]

df_filtered['popularity'].hist(bins=30, figsize=(8, 4), color='coral', edgecolor='white')
plt.title('Distribution of Popularity (excluding 0s)')
plt.xlabel('Popularity (0-100)')
plt.ylabel('Number of songs')
plt.show()
```

> **Answer these questions:**
> 1. Describe the shape of the chart. Is popularity mostly low, high, or spread evenly?
> 2. Does removing the 0-popularity songs change the shape a lot?
> 3. Based on this, do you think predicting popularity will be easy or hard? Why?

---

### Exercise 4 — Explore audio features

**What you're learning:** Understanding what each feature looks like before using it in a model.

```python
import seaborn as sns

# Pick a few interesting features
features = ['danceability', 'energy', 'valence', 'acousticness', 'tempo']

# Plot their distributions side by side
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for i, feature in enumerate(features):
    df[feature].hist(ax=axes[i], bins=30, color='mediumpurple', edgecolor='white')
    axes[i].set_title(feature)
    axes[i].set_xlabel('')

plt.tight_layout()
plt.show()
```

```python
# What are the min/max values for each feature?
print(df[features].describe())
```

> **Answer these questions:**
> 1. Which feature has the most spread-out distribution?
> 2. Look at `acousticness` — is most music in this dataset acoustic or not?
> 3. What does `valence` measure? (hint: check the dataset tab in our project guide)

---

### Exercise 5 — Find correlations with popularity

**What you're learning:** Correlation tells you which features move together. This is the key to knowing which features will be useful for predicting popularity.

```python
# Select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute the correlation matrix
corr = numeric_df.corr()

# Show just the correlations with popularity, sorted
pop_corr = corr['popularity'].drop('popularity').sort_values()
print(pop_corr)
```

```python
# Now draw it as a bar chart
pop_corr.plot(kind='barh', figsize=(8, 6), color='teal')
plt.title('Correlation of each feature with Popularity')
plt.xlabel('Correlation coefficient (-1 to +1)')
plt.axvline(x=0, color='gray', linewidth=0.8)
plt.tight_layout()
plt.show()
```

> **What is a correlation coefficient?**
> - A value between **-1 and +1**
> - Close to **+1** = when this feature goes up, popularity goes up too
> - Close to **-1** = when this feature goes up, popularity goes down
> - Close to **0** = no relationship

> **Answer these questions:**
> 1. Which feature has the highest positive correlation with popularity?
> 2. Which feature has the strongest negative correlation?
> 3. Are any of these correlations very strong (above 0.5 or below -0.5)?

---

### Exercise 6 — Draw the full correlation heatmap

**What you're learning:** See how ALL features relate to each other, not just to popularity.

```python
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr,
    annot=True,       # show the numbers inside the boxes
    fmt='.2f',        # round to 2 decimal places
    cmap='coolwarm',  # red = positive, blue = negative
    center=0,
    square=True,
    linewidths=0.5
)
plt.title('Correlation Heatmap of All Features')
plt.tight_layout()
plt.show()
```

> **Answer these questions:**
> 1. Find `energy` and `loudness` — what is their correlation? Does it make sense?
> 2. Find `energy` and `acousticness` — what is their correlation? Why do you think that is?
> 3. Are there any two features that are so correlated with each other that we might not need both?

---

### Exercise 7 — Explore by genre

**What you're learning:** Some genres are more popular than others. This kind of group-level insight is very useful for a presentation.

```python
# Average popularity per genre (top 15 most popular genres)
genre_popularity = (
    df.groupby('track_genre')['popularity']
    .mean()
    .sort_values(ascending=False)
    .head(15)
)

genre_popularity.plot(kind='bar', figsize=(12, 5), color='salmon', edgecolor='white')
plt.title('Top 15 genres by average popularity')
plt.ylabel('Average popularity score')
plt.xlabel('Genre')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

```python
# How many songs per genre?
df['track_genre'].value_counts().head(20).plot(kind='bar', figsize=(12, 5), color='skyblue')
plt.title('Number of songs per genre (top 20)')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

> **Answer these questions:**
> 1. Which genre has the highest average popularity?
> 2. Is the number of songs per genre balanced or very unequal?
> 3. Name one genre that surprises you in the top 15.

---

### Exercise 8 — Scatter plot: one feature vs. popularity

**What you're learning:** Scatter plots show if there's a visual relationship between two variables.

```python
# Sample 5000 songs (plotting all 114k would be too slow)
sample = df[df['popularity'] > 0].sample(5000, random_state=42)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, feature in enumerate(['danceability', 'energy', 'loudness']):
    axes[i].scatter(sample[feature], sample['popularity'], alpha=0.2, s=5, color='purple')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('popularity')
    axes[i].set_title(f'{feature} vs popularity')

plt.tight_layout()
plt.show()
```

> **Answer these questions:**
> 1. Do any of these scatter plots show a clear upward or downward trend?
> 2. Why do you think the relationship between audio features and popularity might be weak?
> 3. Based on everything so far — which 3 features do you think will be most useful for predicting popularity?

---

## Part 3 — Write a short summary

At the bottom of your notebook, add a new markdown cell (click `+ Markdown`) and write a short summary answering:

```
## My EDA findings

1. Dataset size: ___ rows, ___ columns
2. Missing values: yes / no — in which columns?
3. Duplicate songs: yes / no — how many?
4. Popularity distribution: describe in 1–2 sentences
5. Top 3 features correlated with popularity: ___, ___, ___
6. Most interesting thing I found: ___
7. One question I still have about the data: ___
```

---

## Part 4 — Save and push to GitHub

When you're done, save your notebook and push your work:

```bash
git add notebooks/eda.ipynb
git commit -m "Add EDA notebook: distributions, correlations, genre analysis"
git push origin feature/eda
```

Then go to GitHub and open a **Pull Request** from `feature/eda` into `main`.  
In the PR description, paste your "EDA findings" summary from Part 3.

---

## Checklist before opening your Pull Request

- [ ] Ran all 8 exercises without errors
- [ ] Answered all questions in notebook comments
- [ ] Wrote the EDA findings summary
- [ ] All charts display correctly
- [ ] Committed and pushed to `feature/eda`
- [ ] Opened a Pull Request on GitHub

---

## Need help?

If you get stuck, here are the most common errors and fixes:

| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'pandas'` | Run `pip install pandas` in the terminal |
| `FileNotFoundError: dataset.csv` | Check the file path — make sure csv is in `data/` folder |
| `KeyError: 'popularity'` | Run `df.columns` first to check the exact column name |
| Chart doesn't appear | Add `plt.show()` at the end of your chart code |
| Kernel dies / restarts | Your sample size is too large — reduce to `sample(2000, ...)` |
