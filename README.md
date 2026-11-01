# Naïve Bayes Review Sentiment Analysis

## Files
- `Naive_Bayes_Review_Classification.ipynb`: assignment notebook
- `Reviews_data.csv`: supplied review dataset
- `preprocessing.py`: shared text-cleaning function
- `sentiment_model.pkl`: trained Bernoulli Naïve Bayes pipeline
- `app.py`: Streamlit application
- `requirements.txt`: dependencies

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push `app.py`, `preprocessing.py`, `sentiment_model.pkl`, and `requirements.txt` to GitHub. Then create an app at https://share.streamlit.io/ and set the main file to `app.py`.

The model expects cleaned review text. The Streamlit app calls `clean_text()` before prediction. Keep `preprocessing.py` and `sentiment_model.pkl` together in the repository root.
