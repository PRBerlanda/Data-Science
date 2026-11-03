# 13 · Text Mining, Sentiment Analysis and Text Classification (NLP)

A ready-to-use tool for a **column of texts**: comments, reviews, survey answers, support tickets, e-mails. It cleans the texts, finds the most frequent words and expressions, draws a word cloud, discovers topics with LDA and, when the texts have a label or a rating, trains a text classifier that can label new texts. Lexicon-based sentiment is optional.

> Part of the [Data-Science](https://github.com/PRBerlanda/Data-Science/tree/main) repository. Each branch holds one model.

---

## What this model does

Text mining turns free text into numbers that can be counted and modeled:

- **Cleaning**: lowercase, remove links, numbers, punctuation and stopwords (very common words such as "the", "of"), and optionally reduce words to their stem.
- **Frequencies and bigrams**: which words and two-word expressions appear most.
- **Topics (LDA)**: Latent Dirichlet Allocation groups words that tend to appear together into topics. Each text gets a mix of topics and a dominant one.
- **Classification**: TF-IDF turns each text into a vector that gives more weight to words that are frequent in the text but rare in the dataset. A Naive Bayes and a logistic regression are compared by cross-validation.
- **Lexicon sentiment** (optional): VADER for English, or your own word → polarity list, with simple negation handling.

Built-in stopword lists cover **Portuguese and English**, so the tool works offline.

## When to use it

- Understanding what customers or employees talk about (exploration only: no label needed).
- Classifying texts automatically into known categories: sentiment, ticket subject, complaint type…
- Turning star ratings into positive/negative labels to train a sentiment classifier.

**Limitations:** irony, slang and complex negations are hard for bag-of-words methods. For those cases, consider transformer-based language models.

## How it works

1. **Load the texts** and drop empty ones. Optionally create a label from a rating: high ratings become *positive*, low ratings *negative*, and middle ratings are dropped.
2. **Clean** each text according to the language and options.
3. **Frequent words and bigrams**, bar chart and word cloud.
4. **LDA topics**: the number of topics is chosen automatically (2 to 8) by a simple coherence score, or fixed. Each text receives its dominant topic.
5. **Lexicon sentiment** (if configured): a polarity score and a positive/neutral/negative class per text.
6. **Text classifier** (if there is a label): stratified train/test split, cross-validated comparison of Naive Bayes and logistic regression (macro F1), test evaluation, confusion matrix, most indicative terms per class, and the final model saved after refitting on all data.
7. **Classify new texts** from a file, with a confidence score.

## Inputs

**Data:** one row per text. Optionally a label column or a rating column.

**Configuration** (the `CONFIGURATION` cell):

| Parameter | What it controls |
|---|---|
| `FILE`, `SHEET`, `SEPARATOR` | Input file, Excel sheet and CSV separator. |
| `TEXT_COLUMN` | Column with the texts. |
| `LABEL_COLUMN` | Column with the text class (`None` = exploration only). |
| `LABEL_FROM_RATING` | Build the label from a rating, e.g. `{"column": "review_score", "positive_min": 4, "negative_max": 2}`. |
| `LANGUAGE` | `"portuguese"` or `"english"` (stopwords and stemmer). |
| `REMOVE_ACCENTS`, `EXTRA_STOPWORDS`, `STEMMING`, `MIN_WORD_LENGTH` | Cleaning options. |
| `NGRAMS` | N-gram range used by the classifier, e.g. `(1, 2)` = words and bigrams. |
| `TOP_N`, `WORD_CLOUD` | Number of top terms reported and the word cloud. |
| `N_TOPICS` | Number of LDA topics, `"auto"` or 0 to turn off. |
| `FOLDS`, `TEST_PROPORTION`, `SEED` | Classifier validation settings. |
| `PREDICT_FILE` | New texts to classify (same text column). |
| `SENTIMENT_LEXICON` | `None`, `"vader"` (English) or the path to a CSV with word and polarity columns. |

## Outputs

Saved in `results/`:

| File | Content |
|---|---|
| `report.md` | Text statistics, cleaning example, top terms, topics, sentiment, classifier results and interpretation. |
| `processed_texts.csv` | Original data plus clean text, dominant topic and sentiment (when computed). |
| `frequent_words.csv`, `frequent_bigrams.csv` | Top words and expressions with their frequencies. |
| `lda_topics.csv` | The ten keywords of each topic. |
| `text_classifier.joblib` | Trained TF-IDF + classifier pipeline and its cleaning settings. |
| `new_text_predictions.csv` | Predicted class and confidence for `PREDICT_FILE`. |
| `01_frequent_words.png`, `02_word_cloud.png`, `03_confusion_matrix.png` | Charts. |

## How to read the results

- **Frequent words and bigrams** show the recurring themes. Bigrams often carry more meaning ("long wait", "friendly staff").
- **Topics**: read the keywords of each topic and give it a name. The distribution of dominant topics shows which themes prevail.
- **Lexicon sentiment**: a positive score means positive tone. This is a rough measure that misses irony and context.
- **Classifier**: macro F1 balances all classes (1 = perfect). Trust the classifier only if the cross-validated F1 is stable (small standard deviation) and close to the test F1.
- **Confusion matrix**: which classes the classifier mixes up.
- **Indicative terms**: the words that pull most towards each class. If they make no sense, the classifier is learning noise or the labels need review.
- **Confidence** of a new prediction: the highest class probability. Low values deserve a human review.

## How to run

```bash
pip install -r requirements.txt
```

1. Put your data file next to `text_mining_nlp.py`.
2. Edit the `CONFIGURATION` cell (at least `FILE`, `TEXT_COLUMN` and `LANGUAGE`).
3. Run the script in Spyder (F5) or with `python text_mining_nlp.py`.
4. Open `results/report.md`.

`nltk` (stemming, VADER) and `wordcloud` are optional.

## Example dataset

`depoimentos.csv` contains restaurant customer reviews written in Portuguese: an ID, a date (`data`) and the review text (`depoimento`). It has no label, so the default configuration runs the exploration part (frequencies, word cloud and topics).

## Credits

Built from the "text_mining_1/2/3_colab" notebooks, Classes 01–03 of the Text Mining module (MBA in Data Science and Analytics, USP/Esalq).
