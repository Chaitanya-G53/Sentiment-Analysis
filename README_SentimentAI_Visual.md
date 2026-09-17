# 🎬 SentimentAI

### IMDb Movie Review Sentiment Analysis \| NLP • Machine Learning • Streamlit

```{=html}
<p align="center">
```
`<b>`{=html}Turn natural-language movie reviews into instant sentiment
predictions.`</b>`{=html}
```{=html}
</p>
```
```{=html}
<p align="center">
```
`<a href="https://dmekvxnyfups4ouvwbkjs4.streamlit.app/">`{=html}🌐 Live
Demo`</a>`{=html} •
`<a href="https://github.com/Chaitanya-G53/Sentiment-Analysis">`{=html}💻
GitHub`</a>`{=html} •
`<a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews">`{=html}📊
Dataset`</a>`{=html}
```{=html}
</p>
```

------------------------------------------------------------------------

## 📌 Overview

**SentimentAI** is an end-to-end Natural Language Processing project
that analyzes movie reviews and predicts whether the expressed sentiment
is **Positive** or **Negative**.

The project covers the complete machine-learning workflow:

**Raw Text → Preprocessing → TF-IDF → Model Training → Model
Serialization → Deployment**

The trained model is integrated into an interactive **Streamlit web
application**, allowing users to enter their own review and receive a
real-time prediction.

------------------------------------------------------------------------

## ✨ Project Highlights

  -------------------- ---------------------------------
  📝 **Input**         Natural-language movie review
  🧹 **NLP**           Text cleaning & normalization
  🔢 **Features**      TF-IDF, 5,000 features
  🤖 **Model**         Multinomial Naive Bayes
  📊 **Task**          Binary sentiment classification
  💾 **Persistence**   Pickle
  🌐 **Deployment**    Streamlit
  📚 **Dataset**       IMDb 50K Movie Reviews
  -------------------- ---------------------------------

------------------------------------------------------------------------

## 🚀 Live Application

### Try it yourself

```{=html}
<p align="center">
```
**👉 [OPEN SENTIMENTAI](https://dmekvxnyfups4ouvwbkjs4.streamlit.app/)**

```{=html}
</p>
```
Enter a movie review and the application processes the text through the
saved NLP pipeline before displaying the predicted sentiment and
confidence value.

------------------------------------------------------------------------

## 🎥 Demo

**Project Demo Video:** `Sentiment-Analysis.mp4`

The demo shows the deployed Streamlit application processing a review
and returning its sentiment prediction.

> 📌 **For GitHub:** Upload the video to your repository (for example,
> `assets/Sentiment-Analysis.mp4`) and link it here if GitHub video
> preview is desired.

------------------------------------------------------------------------

## 🧠 How It Works

``` text
                    USER REVIEW
                         │
                         ▼
                ┌─────────────────┐
                │ Text Preprocess  │
                │ • HTML removal   │
                │ • Regex cleaning │
                │ • Lowercasing    │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  TF-IDF Vector  │
                │  5,000 Features │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Multinomial NB  │
                └────────┬────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │ Sentiment Prediction │
              │ Positive / Negative  │
              │ + Confidence         │
              └──────────────────────┘
```

------------------------------------------------------------------------

## 📊 Dataset

**IMDb Dataset of 50K Movie Reviews**

🔗 **[View dataset on
Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)**

The project uses the Kaggle IMDb movie-review dataset for binary
sentiment classification.

The dataset contains:

-   **50,000 movie reviews**
-   **Positive and Negative sentiment labels**
-   Natural-language review text

------------------------------------------------------------------------

## 🧹 NLP Preprocessing

Before converting text into numerical features, the reviews are cleaned.

### Processing steps

``` text
Raw Review
    ↓
Remove HTML tags
    ↓
Remove non-alphabetic characters / numbers
    ↓
Normalize whitespace
    ↓
Convert to lowercase
    ↓
Clean Review
```

### Example

**Before**

``` text
"This movie was AMAZING!!! <br /><br /> I loved it."
```

**After**

``` text
"this movie was amazing i loved it"
```

------------------------------------------------------------------------

## 🔢 Feature Engineering --- TF-IDF

The cleaned review text is transformed into numerical features using
**TF-IDF (Term Frequency--Inverse Document Frequency)**.

The implementation uses:

``` python
TfidfVectorizer(max_features=5000)
```

This converts textual information into a numerical representation that
can be processed by the machine-learning classifier.

------------------------------------------------------------------------

## 🤖 Machine Learning

### Model Used

**Multinomial Naive Bayes**

``` python
MultinomialNB()
```

The project uses an **80/20 train-test split** with `random_state=42`.

``` text
80% → Training
20% → Testing
```

The model learns patterns in the TF-IDF representation and predicts the
sentiment of previously unseen reviews.

------------------------------------------------------------------------

## 📈 Evaluation

The project evaluates the classifier using classification metrics such
as:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   Classification report

> **Note:** Final performance numbers are intentionally not hard-coded
> in this README. Add the exact metrics from your final trained model
> output rather than reporting numbers from a different experiment.

------------------------------------------------------------------------

## 💾 Model Deployment Pipeline

After training, the model and vectorizer are serialized so the Streamlit
application can load them directly.

``` text
Training
   │
   ├── TF-IDF Vectorizer
   │        ↓
   │   vectorizer.pkl
   │
   └── Naive Bayes Model
            ↓
      Sentiment_analysis.pkl
```

During prediction:

``` text
User Text
   ↓
Preprocessing
   ↓
Saved Vectorizer
   ↓
Saved Model
   ↓
Prediction
   ↓
Confidence
```

------------------------------------------------------------------------

## 🛠️ Tech Stack

**Languages & Libraries**

`Python` · `Pandas` · `NumPy` · `NLTK` · `Scikit-learn`

**NLP**

`Text Preprocessing` · `TF-IDF`

**Machine Learning**

`Multinomial Naive Bayes` · `Classification`

**Deployment**

`Streamlit`

**Model Persistence**

`Pickle`

**Dataset**

`Kaggle · IMDb 50K Movie Reviews`

------------------------------------------------------------------------

## 📁 Project Structure

``` text
Sentiment-Analysis/
│
├── 📓 SentimentAnalysis.ipynb
├── 🐍 app.py
├── 🤖 Sentiment_analysis.pkl
├── 🔢 vectorizer.pkl
├── 📦 requirements.txt
├── 📄 README.md
│
└── 🎥 assets/
    └── Sentiment-Analysis.mp4
```

------------------------------------------------------------------------

## 🧪 Example Prediction

### Input

``` text
I absolutely loved this movie. The story was engaging
and the performances were excellent.
```

### Application Output

``` text
Sentiment : Positive
Confidence: Displayed by the application
```

You can test the application with both positive and negative reviews
through the live demo.

------------------------------------------------------------------------

## ⚠️ Limitations

This model is trained specifically on **IMDb movie reviews**.

Therefore, its predictions may not generalize perfectly to:

-   Social-media posts
-   Product reviews
-   Restaurant reviews
-   Very short messages
-   Sarcastic statements
-   Text from domains very different from movie reviews

The preprocessing also removes some non-alphabetic information, which
may discard potentially useful sentiment signals.

------------------------------------------------------------------------

## 🔮 Future Improvements

-   [ ] Compare Naive Bayes with Logistic Regression
-   [ ] Compare with Linear SVM
-   [ ] Tune TF-IDF parameters and n-grams
-   [ ] Add confusion-matrix visualization
-   [ ] Perform systematic error analysis
-   [ ] Experiment with word embeddings
-   [ ] Compare classical ML with transformer-based NLP
-   [ ] Add batch CSV prediction
-   [ ] Add prediction analytics/dashboard
-   [ ] Improve handling of sarcasm and contextual sentiment

------------------------------------------------------------------------

## 🔗 Project Links

  ---------------------------------------------------------------------------------------------------------------------------------
  Resource                            Link
  ----------------------------------- ---------------------------------------------------------------------------------------------
  🌐 **Live Demo**                    [Streamlit App](https://dmekvxnyfups4ouvwbkjs4.streamlit.app/)

  💻 **Source Code**                  [GitHub Repository](https://github.com/Chaitanya-G53/Sentiment-Analysis)

  📊 **Dataset**                      [Kaggle --- IMDb 50K
                                      Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
  ---------------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## 👤 Author

### Chaitanya Girhepunje

Interested in **Machine Learning, NLP, Data Science and AI-based
applications**.

🔗 [GitHub](https://github.com/Chaitanya-G53)

------------------------------------------------------------------------

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on
GitHub.

**Built with Python • NLP • Machine Learning • Streamlit**
