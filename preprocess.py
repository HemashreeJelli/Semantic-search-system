import re
import os
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

def clean_text(text):
    if not isinstance(text, str):
        return ""

    # Remove PGP signatures
    text = re.sub(r'-----BEGIN PGP SIGNATURE-----[\s\S]*?-----END PGP SIGNATURE-----', '', text)

    # Remove common email signature blocks
    text = re.sub(r'--\s*\n[\s\S]*', '', text)

    return text.strip()


# Load dataset
print("Loading 20 Newsgroups dataset...")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'))

df = pd.DataFrame({
    "text": newsgroups.data,
    "label": newsgroups.target
})

# Clean text
print("Cleaning text...")
df["text"] = df["text"].apply(clean_text)

# Remove very small documents
df = df[df["text"].str.len() > 20].reset_index(drop=True)

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Save cleaned dataset
df.to_csv("data/cleaned_20ng.csv", index=False)

print("Saved cleaned dataset to data/cleaned_20ng.csv")
print("Final dataset size:", len(df))