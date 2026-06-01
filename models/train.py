import pandas as pd
import joblib, os, sys
import re

from urllib.parse import urlparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from analyze import extract_features, FEATURE_NAMES
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
#  URL Cleaning
def clean_url(url):
    if pd.isna(url): return None
    url = str(url).strip().lower()
    if url in ["", "nan", "none", "null"]: return None
 
    # Remove internal whitespace
    url = re.sub(r"\s+", "", url)
 
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
 
    # Validate IPv6 brackets
    if url.count("[") != url.count("]"):
        return None
 
    try:
        parsed = urlparse(url)
        if not parsed.netloc: return None
        if "." not in parsed.netloc: return None
        return url
    except:
        return None
 
# Load Dataset 
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'phishing_site_urls.csv')
 
print("📂 Loading dataset...")
df = pd.read_csv(DATASET_PATH, dtype=str, keep_default_na=False, encoding='utf-8')
df = df.drop_duplicates().dropna()
df["Label"] = df["Label"].str.lower().map({"good": 0, "bad": 1})
df = df.dropna(subset=["Label"])
 
df_good = df[df["Label"] == 0].copy()
df_bad  = df[df["Label"] == 1].copy()
 
print(f"✅ Safe: {len(df_good):,} | Malicious: {len(df_bad):,}")
 
# Undersample Safe Class to Balance Dataset
TARGET_GOOD = min(350000, len(df_good))
df_good = df_good.sample(n=TARGET_GOOD, random_state=42)
print(f"⚖️  After balancing - Safe: {len(df_good):,} | Malicious: {len(df_bad):,}")
 
# Clean URLs 
df_good["clean_url"] = df_good["URL"].apply(clean_url)
df_bad["clean_url"]  = df_bad["URL"].apply(clean_url)
 
df_good = df_good[df_good["clean_url"].notna()].reset_index(drop=True)
df_bad  = df_bad[df_bad["clean_url"].notna()].reset_index(drop=True)
 
print(f"✅ After cleaning - Safe: {len(df_good):,} | Malicious: {len(df_bad):,}")
 
# ─── Feature Extraction 
print("⚙️  Extracting features...")
 
good_features = []
for url in df_good["clean_url"]:
    f = extract_features(url, 0)
    if f is not None:
        good_features.append(f)
 
bad_features = []
for url in df_bad["clean_url"]:
    f = extract_features(url, 1)
    if f is not None:
        bad_features.append(f)
 
good = pd.DataFrame(good_features, columns=FEATURE_NAMES)
bad  = pd.DataFrame(bad_features,  columns=FEATURE_NAMES)
 
# ─── Merge and Shuffle ────────────────────────────────────────────
data = pd.concat([good, bad]).sample(frac=1, random_state=42).reset_index(drop=True)
 
X = data.drop(columns=["result"])
y = data["result"]
 
print(f"📊 Total samples: {len(data):,} | Safe: {int(y.value_counts()[0]):,} | Malicious: {int(y.value_counts()[1]):,}")
print(f"📊 Class distribution:\n{y.value_counts(normalize=True).round(3)}")
 
# ─── Train/Test Split ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# ─── Train Model with Pipeline + StandardScaler ───────────────────
print(" Training model...")
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        class_weight={0: 1, 1: 2},
        n_jobs=-1
    ))
])
 
pipeline.fit(X_train, y_train)
 
# ─── Evaluate Model ───────────────────────────────────────────────
print(f"\n Train Accuracy: {pipeline.score(X_train, y_train)*100:.2f}%")
print(f" Test Accuracy:  {pipeline.score(X_test, y_test)*100:.2f}%")
 
y_pred = pipeline.predict(X_test)
print(f"\n{classification_report(y_test, y_pred, target_names=['SAFE', 'MALICIOUS'])}")
 
# ─── Confusion Matrix ─────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"  Safe → Safe:       {cm[0][0]:,}")
print(f"  Safe → Malicious:  {cm[0][1]:,}")
print(f"  Malicious → Safe:  {cm[1][0]:,}  ← Risk")
print(f"  Malicious → Malicious: {cm[1][1]:,}  ← Correct")

# ─── Top Features ─────────────────────────────────────────────────
feature_names = [f for f in FEATURE_NAMES if f != "result"]
importances = pd.Series(
    pipeline.named_steps['classifier'].feature_importances_,
    index=feature_names
)
print("\n🔍 Top 10 Features:")
print(importances.nlargest(10).to_string())
 
# ─── Save Model ───────────────────────────────────────────────────
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'linkshield_model.pkl')
joblib.dump(pipeline, out)
print(f"\n✅ Model saved to: {out}")