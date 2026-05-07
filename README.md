# Data Analysis & Machine Learning Project: Decomposing Toxic Language

**Group Members:** Itgel Enkh-Amgalan, Zan Zhang 

## Research Question
How do lexical signals, structural writing signals, and their combination participate in classifying online toxicity subtypes?

## Dataset
The project uses the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge) dataset from Kaggle. It contains Wikipedia comments labeled across six toxicity subtypes: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. The dataset is heavily imbalanced — roughly 90% of comments are non-toxic.

## Project Structure
The notebook is organized into the following sections:

1. **Data Import** — loads the dataset via the Kaggle API
2. **Cleaning & Preprocessing** — removes noise, strips whitespace, and filters out non-alphabetic comments
3. **EDA & Visualization** — explores label distribution, multi-label overlap, correlations, comment length, and structural feature patterns
4. **Model Building** — trains three logistic regression baselines per toxicity label: lexical-only (TF-IDF), structural-only, and combined
5. **Model Evaluation** — compares models using F1-score and PR-AUC
6. **Structural Feature Coefficient Evaluation** — identifies which structural features contribute most as supporting signals, backed by Mann-Whitney U and chi-square statistical tests
7. **Comment Classifier Demo** — interactive terminal-based classifier using the trained combined models

## Methods
Three logistic regression baselines are trained for each of the six toxicity labels:

- **Lexical-only:** TF-IDF with unigrams and bigrams (`max_features=5000`)
- **Structural-only:** handcrafted features including comment length, word count, uppercase ratio, punctuation ratio, and repeated letter/punctuation counts
- **Combined:** lexical and structural features concatenated into a single feature matrix

All models use `class_weight='balanced'` to handle class imbalance, and structural features are standardized using `StandardScaler` before training.

## Key Findings
- Lexical features are the dominant signal — TF-IDF alone accounts for most of the model's predictive power across all six labels
- Structural features are weak on their own and add only marginal improvement when combined with lexical features
- `toxic`, `obscene`, and `insult` perform comparably well due to shared vocabulary, while `threat` and `identity_hate` are the hardest labels to classify
- `uppercase_ratio` is the most consistent structural signal across toxicity subtypes; repeated-letter and punctuation features contribute very little

## How to Run
1. Set up the Kaggle API and authenticate via `kagglehub`
2. The dataset can be loaded in two ways:
   - **From Kaggle directly:** use `kagglehub.competition_download()` to fetch the data at runtime
   - **From Google Drive:** save the dataset locally after the first download and load it from your Drive path to avoid re-downloading every session
3. Run all cells from top to bottom (choose how you load the dataset) — the demo classifier at the end requires the trained model objects to be in memory
4. When the demo cell runs, type a comment into the terminal prompt to classify it, or type `quit` to exit
## Dependencies
```
pandas
numpy
scikit-learn
scipy
matplotlib
kagglehub
```
