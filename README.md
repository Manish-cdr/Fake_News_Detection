# Fake News Detection

A small web app that reads a news article and predicts whether it is **Real** or **Fake**, with a confidence percentage. It uses TF-IDF features and logistic regression, served through Flask.

## Project structure

```
fake-news-detection/
├── app.py              # Flask web app
├── train_model.py      # Cleans data, trains and saves the model
├── preprocess.py       # Removes "Reuters" give-aways from text
├── Fake.csv            # Fake articles (not included in the tuned zip)
├── True.csv            # Real articles (not included in the tuned zip)
├── model/
│   └── model.pkl       # Trained model (created by train_model.py)
├── templates/
│   └── index.html      # Web page
└── static/
    └── style.css       # Styling
```

## Setup

Requires Python 3.9+.

```bash
pip install flask scikit-learn pandas joblib
```

Place `Fake.csv` and `True.csv` in the project folder (next to `train_model.py`).

## Train the model

```bash
python train_model.py
```

This prints a classification report (precision, recall, F1) on a held-out test set and saves the model to `model/model.pkl`.

## Run the app

```bash
python app.py
```

Open http://127.0.0.1:5000, paste an article, and click **Check**.

## How it works

1. **Load and label** the two CSVs (Fake = 0, Real = 1).
2. **Clean the data**
   - Drop exact duplicate articles, which would otherwise leak between train and test sets.
   - Drop articles with empty text.
   - Remove the word "Reuters" and the "CITY (Reuters) -" dateline (see below).
3. **Combine** title and text into one field.
4. **Split** 80% train / 20% test, stratified by label.
5. **Train** a pipeline of `TfidfVectorizer` followed by `LogisticRegression`.

### Why Reuters is removed

Almost every article in `True.csv` mentions "Reuters", while very few in `Fake.csv` do. Left in, the model can score well just by spotting that word, which does not help on articles from other sources. `preprocess.py` strips it, and because it is part of the saved pipeline, the web app applies the same cleaning automatically.

## Tuned parameters

| Component | Parameter | Original | Tuned |
|---|---|---|---|
| TfidfVectorizer | `stop_words` | `"english"` | removed |
| TfidfVectorizer | `sublinear_tf` | off | `True` |
| TfidfVectorizer | `min_df` | 1 | `2` |
| TfidfVectorizer | `max_df` | 1.0 | `0.95` |
| TfidfVectorizer | `preprocessor` | none | `preprocess` (removes Reuters, lowercases) |
| TfidfVectorizer | `ngram_range` | `(1, 1)` | `(1, 1)` |
| LogisticRegression | `C` | 1 | `100` |
| LogisticRegression | `solver` | `lbfgs` | `liblinear` |
| LogisticRegression | `max_iter` | 100 | `1000` |

Parameters were chosen on a validation split taken from the training data, so the test set was not used for tuning.

## Results

Measured on the same cleaned data and test split:

| Setup | Accuracy | F1 (Real) |
|---|---|---|
| Original settings | 97.67% | 0.9789 |
| Tuned settings | **99.11%** | **0.9919** |

## Limitations

- The dataset comes from a small number of sources with very different writing styles, so a high score here does not guarantee the same accuracy on news from other websites.
- The model judges writing style and vocabulary. It does not check facts.
- Test it on articles from outside this dataset before relying on it.

## Notes

- If you change `preprocess.py` or retrain, restart `app.py` so it loads the new model.
- `app.py` runs with `debug=True`, which is fine for local use but should be turned off before any deployment.
