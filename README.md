SentimentAI --- IMDb Movie Review Sentiment Analysis

A Natural Language Processing (NLP) project that classifies movie-review
text as Positive or Negative using TF-IDF feature extraction and
a Multinomial Naive Bayes classifier.

🚀 Live Demo

Try the deployed Streamlit application:

Open SentimentAI

The application allows a user to enter review text and returns: -
Predicted sentiment: Positive / Negative - A displayed confidence
score - A simple interactive interface for real-time prediction

📂 GitHub Repository

View the complete project on
GitHub

🎥 Demo Video

A demo video is included with the project materials. It demonstrates the
deployed application using a real-world review example and shows the
sentiment prediction and confidence score.

If you upload the video to this repository, place it at
assets/Sentiment-Analysis.mp4 and add a repository link here.

📊 Dataset

This project uses the IMDB Dataset of 50K Movie Reviews from Kaggle.

View / Download the Kaggle
dataset

The dataset contains movie reviews labelled as positive or
negative. The Kaggle dataset describes 50,000 reviews and a binary
sentiment-classification task.

🧠 Project Workflow

Movie Review
     ↓
Text Cleaning
     ↓
TF-IDF Vectorization
     ↓
Multinomial Naive Bayes
     ↓
Sentiment Prediction
     ↓
Streamlit Web App

🔧 Technologies Used

Python

Pandas

NumPy

Regular Expressions (re)

NLTK

Scikit-learn

TF-IDF Vectorization

Multinomial Naive Bayes

Pickle

Streamlit

Kaggle Dataset

🧹 Text Preprocessing

The notebook applies the following cleaning steps:

Removes HTML tags from reviews

Removes non-alphabetic characters and numbers

Normalizes extra whitespace

Converts text to lowercase

Example:

Raw:
"This movie was AMAZING!!! <br /><br /> I loved it."

After cleaning:
"this movie was amazing i loved it"

🔢 Feature Extraction

The cleaned reviews are converted into numerical features using:

TfidfVectorizer(max_features=5000)

The project uses the 5,000 most relevant TF-IDF features for model
training.

🤖 Machine Learning Model

The classifier used in the notebook is:

MultinomialNB()

The vectorized dataset is split into:

80% training data

20% testing data

with:

train_test_split(
    x_vec,
    y,
    test_size=0.2,
    random_state=42
)

💾 Model Serialization

The trained model and TF-IDF vectorizer are saved separately using
Pickle:

Sentiment_analysis.pkl
vectorizer.pkl

This allows the deployed application to load the trained artifacts and
perform predictions without retraining the model for every user request.

🌐 Deployment

The project is deployed using Streamlit, providing an interactive
web interface for sentiment prediction.

Application flow

User enters review
        ↓
Preprocessing
        ↓
Saved TF-IDF vectorizer
        ↓
Saved Naive Bayes model
        ↓
Positive / Negative prediction
        ↓
Confidence displayed

📁 Suggested Repository Structure

Sentiment-Analysis/
│
├── SentimentAnalysis.ipynb
├── app.py
├── Sentiment_analysis.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
│
└── assets/
    └── Sentiment-Analysis.mp4

🧪 Example

Input

I absolutely loved this movie. The story was engaging and the performances were excellent.

Output

Sentiment: Positive
Confidence: [shown by the application]

The application can also be tested with negative reviews to observe the
opposite classification.

📌 Important Project Limitations

This model is trained on IMDb movie reviews, so its performance may
not transfer directly to every type of text.

For example, a model trained on movie reviews may behave differently
on: - Restaurant reviews - Product reviews - Social-media posts - Short
messages - Sarcastic statements

The preprocessing in the notebook also removes non-alphabetic
characters, which can remove information such as punctuation, numbers,
or other sentiment-related signals.

These limitations are important when interpreting predictions from the
deployed application.

🔮 Possible Future Improvements

Compare Multinomial Naive Bayes with Logistic Regression and Linear
SVM

Tune TF-IDF parameters such as ngram_range and min_df

Add stop-word and lemmatization experiments

Add a confusion matrix and detailed evaluation dashboard

Perform error analysis on misclassified reviews

Experiment with word embeddings

Compare the traditional ML approach with transformer-based models
such as BERT

Add batch prediction through CSV upload

Add prediction history and analytics

📚 Dataset Reference

Dataset:

IMDB Dataset of 50K Movie Reviews

Kaggle:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The underlying large movie-review sentiment benchmark is associated with
the work by Andrew Maas and colleagues on learning word vectors for
sentiment analysis.

👤 Author

Chaitanya Girhepunje

GitHub:
https://github.com/Chaitanya-G53

Project Repository:
https://github.com/Chaitanya-G53/Sentiment-Analysis

⭐ If you find this project useful

Feel free to explore the repository, try the live application, and
suggest improvements.
