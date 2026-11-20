# Amazon Search Queries (ESCI) - CMPS261 Project

Mazen Hachem, Ram Haidar, Karim Nabaa

We predict if a product is Exact, Substitute, Complement or Irrelevant for a search query, using the Amazon Shopping Queries dataset.

## Notebook

Everything is in `Amazon_Search_Queries.ipynb`:

- Setup and Part 1: data and preprocessing (Karim)
- Parts 2-4: baselines (TF-IDF + logistic regression, gradient boosted trees), fine-tuned BERT model (MiniLM) and the test comparison (Mazen)
- Part 5: final analyses: checkpoint, complexity, error analysis, ablation, learning curve, MiniLM on all training queries and calibration (Ram)

## Results (test set, 4,000 queries)

| Model | Macro-F1 |
|---|---|
| Always predict Exact | 0.196 |
| Gradient boosted trees | 0.373 |
| TF-IDF + logistic regression | 0.424 |
| MiniLM, 20k training queries | 0.538 |
| MiniLM, all training queries (final model) | 0.563 |

## How to run

Open the notebook in Google Colab (File > Open notebook > GitHub), set the runtime to T4 GPU and do Run all. The first cell downloads the data (around 1.2GB).

On Kaggle: import the notebook, set the accelerator to GPU T4 and turn Internet on, then use Save Version > Save & Run All. The submitted outputs come from a Kaggle run (about 3 hours).

To run it locally: `pip install -r requirements.txt` and run the notebook.

## Models

The trained models will be uploaded in the Releases section (they are too big for the repo).
