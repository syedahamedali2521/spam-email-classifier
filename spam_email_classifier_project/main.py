import pandas as pd
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1️⃣ Load Dataset
# =========================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "dataset", "spam.csv")
df = pd.read_csv(data_path)


# Ensure correct column names
df.columns = ["label", "message"]

# =========================
# 2️⃣ Preprocess the Text
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

df["cleaned"] = df["message"].apply(clean_text)

# =========================
# 3️⃣ Split Dataset
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# =========================
# 4️⃣ Vectorize Text (TF-IDF)
# =========================
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# =========================
# 5️⃣ Train Naive Bayes Model
# =========================
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# =========================
# 6️⃣ Evaluate Model
# =========================
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Accuracy: {accuracy:.2f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 7️⃣ Try Some Example Predictions
# =========================
examples = [
    "Congratulations! You won a free iPhone.",
    "Hey, can we meet tomorrow?",
    "Win $1000 now! Click this link to claim your prize.",
    "Please review the attached project report."
]

for msg in examples:
    msg_clean = clean_text(msg)
    msg_vec = vectorizer.transform([msg_clean])
    pred = model.predict(msg_vec)[0]
    label = "🚫 Spam" if pred == "spam" else "✅ Not Spam"
    print(f"\n🧠 Message: {msg}\n🔍 Prediction: {label}")
