# Amazon Search Queries (ESCI) - CMPS261 Project

Mazen Hachem, Ram Haidar, Karim Nabaa

We predict if a product is Exact, Substitute, Complement or Irrelevant for a search query, using the
[Amazon Shopping Queries dataset](https://github.com/amazon-science/esci-data).

## Notebook

Everything is in `Amazon_Search_Queries.ipynb`:

- Part 1: data and preprocessing (Mazen)
- Part 2: baselines, TF-IDF + logistic regression and gradient boosted trees (Ram)
- Part 3: fine-tuned BERT model (MiniLM) (Karim)
- Part 4: comparison of the models

## How to run

Open the notebook in Google Colab (File > Open notebook > GitHub), set the runtime to T4 GPU and do Run all.
The first cell downloads the data (around 1.2GB).

To run it locally: `pip install -r requirements.txt` and run the notebook.

## Models

The trained models will be uploaded in the Releases section (they are too big for the repo).
