# 🧠 Spam Email Classifier – NLP Mini Project

## 📘 Overview
This project uses Natural Language Processing (NLP) to classify messages as spam or not spam using a Naive Bayes classifier.

## 🧩 Steps
1. Clean and preprocess text (remove stopwords, punctuation)
2. Convert text to numerical features using TF-IDF
3. Train a Naive Bayes classifier
4. Test and evaluate accuracy

## ⚙️ Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python main.py
   ```

## 🧠 Example Output
```
✅ Accuracy: 0.97
🧠 Message: Congratulations! You won a free iPhone.
🔍 Prediction: 🚫 Spam
```

## 🧑‍💻 Libraries Used
- pandas
- scikit-learn
- nltk
