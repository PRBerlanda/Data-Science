# -*- coding: utf-8 -*-
"""
TEXT MINING, SENTIMENT ANALYSIS AND TEXT CLASSIFICATION (NLP)
MBA in Data Science and Analytics (USP/Esalq). Working version generated with
Claude from the "text_mining_1/2/3_colab" notebooks (Classes 01–03).

GOAL
- Explore a column of texts (comments, reviews, tickets, e-mails): cleaning,
  most frequent words and expressions, word cloud, topics.
- If there is a LABEL (e.g. positive/negative, ticket subject) or a rating
  (e.g. 1 to 5 stars), trains and evaluates a text CLASSIFIER
  (TF-IDF + Naive Bayes / Logistic Regression) and classifies new texts.
- Optional: lexicon-based sentiment (VADER for English or your own CSV lexicon).

WORKS OFFLINE
- Built-in Portuguese and English stopword lists (NLTK is used, if installed,
  for the RSLP/Porter stemmers). wordcloud is optional.

Installation (once, in the Anaconda Prompt):
    pip install pandas numpy scikit-learn matplotlib seaborn openpyxl joblib
    (optional) pip install nltk wordcloud
"""

#%% CONFIGURATION (the only place to edit)

FILE = "depoimentos.csv"
SHEET = None
SEPARATOR = None
TEXT_COLUMN = "depoimento"
LABEL_COLUMN = None             # column with the text class (None = exploration only)
LABEL_FROM_RATING = None        # e.g. {"column": "review_score",
                                #       "positive_min": 4, "negative_max": 2}
                                # (intermediate ratings are discarded)

# --- Preprocessing --------------------------------------------------------
LANGUAGE = "portuguese"         # "portuguese" or "english"
REMOVE_ACCENTS = False
EXTRA_STOPWORDS = []            # words to ignore besides the default ones
STEMMING = False                # RSLP (pt) / Porter (en); requires nltk
MIN_WORD_LENGTH = 2
NGRAMS = (1, 2)                 # unigrams and bigrams

# --- Exploration ----------------------------------------------------------
TOP_N = 25
WORD_CLOUD = True
N_TOPICS = "auto"               # LDA: integer, "auto" (2 to 8 by simple
                                # coherence) or 0 to turn off

# --- Classification -------------------------------------------------------
FOLDS = 5
TEST_PROPORTION = 0.2
SEED = 42
PREDICT_FILE = None             # new texts to classify (same TEXT_COLUMN)

# --- Lexicon-based sentiment (optional) -----------------------------------
SENTIMENT_LEXICON = None        # None, "vader" (English only, requires nltk) or path
                                # to a CSV with columns word;polarity

OUTPUT_FOLDER = "results"
SHOW_PLOTS = True


#%% Packages

from pathlib import Path
from collections import Counter
import re
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


#%% Utility functions (shared by all tools; no need to edit)

SCRIPT_DIR = (Path(__file__).resolve().parent
              if "__file__" in globals() else Path.cwd())
REPORT = []


def resolve_path(relative):
    """Turn a path relative to the script folder into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def read_data(file, sheet=None, separator=None):
    """
    Read CSV, Excel, Parquet, Pickle or JSON based on the file extension.

    For CSV, the separator is detected and the most common encodings are tried
    (utf-8, utf-8 with BOM and latin-1), avoiding errors with accented text.
    """
    path = resolve_path(file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    ext = path.suffix.lower()
    df = None
    if ext in (".csv", ".txt"):
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, sep=separator, engine="python", encoding=enc)
                break
            except UnicodeDecodeError:
                continue
    elif ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    elif ext == ".parquet":
        df = pd.read_parquet(path)
    elif ext in (".pkl", ".pickle"):
        df = pd.read_pickle(path)
    elif ext == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported extension: {ext}")
    if isinstance(df, pd.Series):
        df = df.to_frame()
    # Remove spaces and invisible marks (BOM) from column names
    df.columns = [str(c).replace("﻿", "").strip() for c in df.columns]
    return df


def prepare_output(folder):
    """Create (if needed) and return the results folder."""
    output = resolve_path(folder)
    output.mkdir(parents=True, exist_ok=True)
    return output


def log(text=""):
    """Print to the console and keep the line for report.md."""
    print(text)
    REPORT.append(str(text))


def save_figure(fig, name):
    """Save the chart as PNG in the output folder and show it (if configured)."""
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{name}.png", dpi=120, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def save_report(title):
    """Write all logged lines to report.md."""
    text = f"# {title}\n\n" + "\n".join(REPORT) + "\n"
    (OUTPUT / "report.md").write_text(text, encoding="utf-8")
    print(f"\nReport saved to: {OUTPUT / 'report.md'}")


def md_table(df, decimals=3):
    """Convert a DataFrame into a Markdown table (without depending on tabulate)."""
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return ""
            if x != 0 and abs(x) < 10 ** -decimals:
                return f"{x:.2e}"
            return f"{round(float(x), decimals):g}" if abs(x) < 1e6 else f"{x:,.0f}"
        return x
    d = df.copy()
    d = d.reset_index()
    if d.columns[0] == "index":
        d = d.rename(columns={"index": ""})
    lines = ["| " + " | ".join(map(str, d.columns)) + " |",
             "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(map(str, r)) + " |"
              for r in d.astype(object).map(fmt).values]
    return "\n".join(lines)


OUTPUT = prepare_output(OUTPUT_FOLDER)


# Portuguese stopword list (language data used when LANGUAGE = "portuguese")
STOPWORDS_PT = set("""
a à ao aos as às até com como da das de dela delas dele deles depois do dos e
é ela elas ele eles em entre era eram essa essas esse esses esta está estão
estas estava estavam este estes eu foi foram há isso isto já lhe lhes mais mas
me mesmo meu meus minha minhas muito na não nas nem no nos nós nossa nossas
nosso nossos num numa o os ou para pela pelas pelo pelos por qual quando que
quem se sem ser será seu seus só sua suas também te tem têm tinha tu tua tuas
um uma umas uns você vocês vos ser sido sou são seja sejam estou estamos
tenho temos tive teve fui fomos vai vão vou pra pro pois porque então assim
aqui ali lá onde cada todo toda todos todas outro outra outros outras sobre
""".split())
# Negation words stay OUT of the stopwords: "não gostei" (didn't like) ≠ "gostei" (liked).
STOPWORDS_PT -= {"não", "nem", "sem"}

STOPWORDS_EN = set("""
a an and are as at be been but by for from had has have he her his i if in
into is it its me my of on or our she so than that the their them they this
to was we were what when which who will with you your
""".split())

# Negation words that flip the polarity of the next word in the lexicon score
NEGATIONS = {"não", "nem", "nunca", "jamais", "not", "no", "never", "nor"}


def prepare_tools():
    """Load stopwords and stemmer (NLTK if available; otherwise, built-in lists)."""
    stop = STOPWORDS_PT if LANGUAGE == "portuguese" else STOPWORDS_EN
    stemmer = None
    if STEMMING:
        try:
            import nltk
            if LANGUAGE == "portuguese":
                try:
                    nltk.data.find("stemmers/rslp")
                except LookupError:
                    nltk.download("rslp", quiet=True)
                stemmer = nltk.stem.RSLPStemmer().stem
            else:
                stemmer = nltk.stem.PorterStemmer().stem
        except Exception as error:
            log(f"- Stemming turned off (nltk unavailable: {error}).")
    return stop | {s.lower() for s in EXTRA_STOPWORDS}, stemmer


STOP, STEMMER = prepare_tools()


def clean(text):
    """Lowercase, remove links/numbers/punctuation and stopwords, (optionally) stem."""
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", t)
    if REMOVE_ACCENTS:
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-zà-ÿ\s]", " ", t)
    words = [w for w in t.split()
             if len(w) >= MIN_WORD_LENGTH and w not in STOP]
    if STEMMER:
        words = [STEMMER(w) for w in words]
    return " ".join(words)


#%% 1. Load texts

data = read_data(FILE, SHEET, SEPARATOR)
if TEXT_COLUMN not in data.columns:
    raise KeyError(f"Text column '{TEXT_COLUMN}' not found: {list(data.columns)}")
data = data[data[TEXT_COLUMN].notna() & (data[TEXT_COLUMN].astype(str).str.strip() != "")]
data = data.reset_index(drop=True)

if LABEL_FROM_RATING:
    cfg = LABEL_FROM_RATING
    rating = pd.to_numeric(data[cfg["column"]], errors="coerce")
    data["label"] = np.where(rating >= cfg["positive_min"], "positive",
                             np.where(rating <= cfg["negative_max"], "negative", None))
    data = data.dropna(subset=["label"]).reset_index(drop=True)
    label_column = "label"
else:
    label_column = LABEL_COLUMN

data["clean_text"] = data[TEXT_COLUMN].map(clean)
data["n_words"] = data[TEXT_COLUMN].astype(str).str.split().str.len()
data = data[data["clean_text"].str.len() > 0].reset_index(drop=True)

log("## 1. Texts")
log(f"- File: `{FILE}` | column: `{TEXT_COLUMN}` | valid texts: {len(data)}")
log(f"- Words per text: median {data['n_words'].median():.0f}, "
    f"maximum {data['n_words'].max()}")
log(f"- Language: {LANGUAGE} | stemming: {'yes' if STEMMER else 'no'} | "
    f"n-grams: {NGRAMS}")
log("- Cleaning example: "
    f"\"{str(data[TEXT_COLUMN].iloc[0])[:120]}\" → \"{data['clean_text'].iloc[0][:120]}\"")


#%% 2. Word and expression frequency

counts = Counter(" ".join(data["clean_text"]).split())
top_words = pd.DataFrame(counts.most_common(TOP_N), columns=["word", "frequency"])
bigrams = CountVectorizer(ngram_range=(2, 2), min_df=2).fit(data["clean_text"])
bigram_freq = np.asarray(bigrams.transform(data["clean_text"]).sum(axis=0)).ravel()
top_bigrams = (pd.DataFrame({"expression": bigrams.get_feature_names_out(),
                             "frequency": bigram_freq})
               .sort_values("frequency", ascending=False).head(TOP_N))

log("\n## 2. Most frequent terms")
log(md_table(top_words.head(15).set_index("word"), 0))
log("\n### Most frequent expressions (bigrams)")
log(md_table(top_bigrams.head(10).set_index("expression"), 0))

fig, ax = plt.subplots(figsize=(8, 0.28 * len(top_words) + 1.5))
ax.barh(top_words["word"][::-1], top_words["frequency"][::-1], color="steelblue")
ax.set(title=f"{TOP_N} most frequent words", xlabel="Frequency", ylabel="Word")
save_figure(fig, "01_frequent_words")

if WORD_CLOUD:
    try:
        from wordcloud import WordCloud
        cloud = WordCloud(width=1000, height=500, background_color="white",
                          colormap="viridis").generate_from_frequencies(counts)
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.imshow(cloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title("Word cloud")
        save_figure(fig, "02_word_cloud")
    except ImportError:
        log("- Word cloud not generated (install: pip install wordcloud).")


#%% 3. Topics (LDA)
# LDA groups words that tend to appear together into "topics". Each text
# gets a mixture of topics; the dominant topic helps organize the dataset.

topics = None
if N_TOPICS and len(data) >= 20:
    lda_vec = CountVectorizer(max_df=0.9, min_df=2, ngram_range=(1, 1))
    matrix = lda_vec.fit_transform(data["clean_text"])
    vocab = lda_vec.get_feature_names_out()

    def coherence(lda):
        """Mean simple coherence (UMass) of the 10 keywords of each topic."""
        binary = (matrix > 0).astype(int)
        doc_freq = np.asarray(binary.sum(axis=0)).ravel()
        scores = []
        for comp in lda.components_:
            top = comp.argsort()[::-1][:10]
            s = 0
            for i in range(1, len(top)):
                for j in range(i):
                    co = binary[:, top[i]].multiply(binary[:, top[j]]).sum()
                    s += np.log((co + 1) / max(doc_freq[top[j]], 1))
            scores.append(s)
        return np.mean(scores)

    candidates = range(2, 9) if N_TOPICS == "auto" else [int(N_TOPICS)]
    evaluations = {}
    for k in candidates:
        lda = LatentDirichletAllocation(n_components=k, random_state=SEED,
                                        learning_method="batch").fit(matrix)
        evaluations[k] = (coherence(lda), lda)
    k_final = max(evaluations, key=lambda k: evaluations[k][0])
    lda = evaluations[k_final][1]
    topics = pd.DataFrame({f"topic_{i + 1}": [vocab[j] for j in comp.argsort()[::-1][:10]]
                           for i, comp in enumerate(lda.components_)})
    data["dominant_topic"] = lda.transform(matrix).argmax(axis=1) + 1
    log(f"\n## 3. Topics (LDA, k = {k_final}"
        f"{', chosen by coherence' if N_TOPICS == 'auto' else ''})")
    log(md_table(topics.T.rename(columns=lambda c: f"w{c + 1}").iloc[:, :8], 0))
    log("- Texts by dominant topic: "
        + str(data["dominant_topic"].value_counts().sort_index().to_dict()))


#%% 4. Lexicon-based sentiment (optional)

if SENTIMENT_LEXICON:
    if SENTIMENT_LEXICON == "vader":
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
        analyzer = SentimentIntensityAnalyzer()
        data["lexicon_sentiment"] = data[TEXT_COLUMN].map(
            lambda t: analyzer.polarity_scores(str(t))["compound"])
    else:
        lex = read_data(SENTIMENT_LEXICON)
        polarity = dict(zip(lex.iloc[:, 0].astype(str).str.lower(),
                            pd.to_numeric(lex.iloc[:, 1], errors="coerce")))

        def score(text):
            words = str(text).lower().split()
            total, flip = 0.0, False
            for w in words:
                if w in NEGATIONS:
                    flip = True
                    continue
                v = polarity.get(w, 0)
                total += -v if flip else v
                flip = False
            return total

        data["lexicon_sentiment"] = data[TEXT_COLUMN].map(score)
    data["lexicon_class"] = np.sign(data["lexicon_sentiment"]).map(
        {1: "positive", 0: "neutral", -1: "negative"})
    log("\n## 4. Lexicon-based sentiment")
    log("- Distribution: " + str(data["lexicon_class"].value_counts().to_dict()))


#%% 5. Text classifier (when there is a label)
# TF-IDF gives high weight to words that are frequent in the text and rare in
# the dataset. The vectorizer stays INSIDE the pipeline: the vocabulary is
# learned only on the training data.

final_model = None
if label_column:
    base = data.dropna(subset=[label_column])
    y = base[label_column].astype(str)
    class_counts = y.value_counts()
    rare = class_counts[class_counts < FOLDS].index.tolist()
    if rare:
        log(f"- Classes with fewer than {FOLDS} examples removed: {rare}")
        base, y = base[~y.isin(rare)], y[~y.isin(rare)]
    X_tr, X_te, y_tr, y_te = train_test_split(base["clean_text"], y,
                                              test_size=TEST_PROPORTION,
                                              random_state=SEED, stratify=y)
    candidates = {
        "naive_bayes": Pipeline([("tfidf", TfidfVectorizer(ngram_range=NGRAMS, min_df=2,
                                                           sublinear_tf=True)),
                                 ("clf", MultinomialNB())]),
        "logistic_regression": Pipeline([("tfidf", TfidfVectorizer(ngram_range=NGRAMS,
                                                                   min_df=2, sublinear_tf=True)),
                                         ("clf", LogisticRegression(max_iter=3000,
                                                                    class_weight="balanced"))]),
    }
    cv = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    cv_scores = {n: cross_val_score(p, X_tr, y_tr, cv=cv, scoring="f1_macro")
                 for n, p in candidates.items()}
    best = max(cv_scores, key=lambda n: cv_scores[n].mean())
    final_model = candidates[best].fit(X_tr, y_tr)
    pred = final_model.predict(X_te)

    log("\n## 5. Text classifier")
    log(f"- Classes: " + ", ".join(f"{k} ({v})" for k, v in y.value_counts().items()))
    for n, s in cv_scores.items():
        log(f"- {n}: macro F1 ({FOLDS}-fold CV) = {s.mean():.4f} ± {s.std():.4f}")
    log(f"- Best: **{best}** | macro F1 on test = "
        f"{f1_score(y_te, pred, average='macro'):.4f}")
    rep = pd.DataFrame(classification_report(y_te, pred, output_dict=True,
                                             zero_division=0)).T
    log(md_table(rep.drop(columns="support", errors="ignore"), 3))

    labels = sorted(y.unique())
    fig, ax = plt.subplots(figsize=(1.0 * len(labels) + 3, 0.8 * len(labels) + 2.5))
    sns.heatmap(confusion_matrix(y_te, pred, labels=labels), annot=True, fmt="d",
                cmap="Blues", cbar=False, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title="Confusion matrix (test)", xlabel="Predicted", ylabel="Actual")
    save_figure(fig, "03_confusion_matrix")

    # Words that pull most towards each class (coefficients/log-probabilities)
    vec = final_model.named_steps["tfidf"]
    clf = final_model.named_steps["clf"]
    terms = vec.get_feature_names_out()
    weights = clf.coef_ if hasattr(clf, "coef_") else clf.feature_log_prob_
    log("\n### Most indicative terms for each class")
    if weights.shape[0] == 1:
        order = weights[0].argsort()
        log(f"- **{clf.classes_[1]}**: " + ", ".join(terms[order[::-1][:12]]))
        log(f"- **{clf.classes_[0]}**: " + ", ".join(terms[order[:12]]))
    else:
        for i, c in enumerate(clf.classes_):
            relative = weights[i] - np.delete(weights, i, axis=0).mean(axis=0)
            log(f"- **{c}**: " + ", ".join(terms[relative.argsort()[::-1][:12]]))

    final_model = candidates[best].fit(base["clean_text"], y)
    joblib.dump({"pipeline": final_model, "cleaning_config": {
        "language": LANGUAGE, "stemming": STEMMING, "extra_stopwords": EXTRA_STOPWORDS}},
        OUTPUT / "text_classifier.joblib")


#%% 6. Save results and classify new texts

output_columns = [c for c in data.columns if c not in ("n_words",)]
data[output_columns].to_csv(OUTPUT / "processed_texts.csv", index=False,
                            encoding="utf-8-sig")
top_words.to_csv(OUTPUT / "frequent_words.csv", index=False, encoding="utf-8-sig")
top_bigrams.to_csv(OUTPUT / "frequent_bigrams.csv", index=False, encoding="utf-8-sig")
if topics is not None:
    topics.to_csv(OUTPUT / "lda_topics.csv", index=False, encoding="utf-8-sig")

if PREDICT_FILE and final_model is not None:
    new_data = read_data(PREDICT_FILE)
    cleaned = new_data[TEXT_COLUMN].map(clean)
    new_data["predicted_class"] = final_model.predict(cleaned)
    probs = final_model.predict_proba(cleaned)
    new_data["confidence"] = probs.max(axis=1)
    new_data.to_csv(OUTPUT / "new_text_predictions.csv", index=False, encoding="utf-8-sig")
    log(f"\n- {len(new_data)} new texts classified.")

log("\n## 6. How to interpret")
log("- Frequencies and bigrams show the recurring themes; the topics "
    "organize those themes into groups (give each one a name).")
log("- The classifier is only reliable if the cross-validated F1 is stable; "
    "review the indicative terms to check that they make sense.")
log("- Irony, slang and complex negations remain hard for bag-of-words "
    "methods.")

save_report("Report: Text Mining and NLP")
