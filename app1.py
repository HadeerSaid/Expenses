# app.py
import os
import re
import cv2
import gdown
import pytesseract
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import streamlit as st
from oauth2client.service_account import ServiceAccountCredentials
import gspread
import json

# Path to Tesseract (already pre-installed on Streamlit Cloud)
#pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# === Google Sheets Setup ===
@st.cache_data
def load_data():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_dict = json.loads(st.secrets["GOOGLE_CREDENTIALS"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Medical Rep Expense Submission (Responses)").sheet1
    return pd.DataFrame(sheet.get_all_records())

# === Receipt Processing ===
def download_receipt(url):
    match = re.search(r"id=([a-zA-Z0-9_-]+)", url)
    if not match:
        return None
    file_id = match.group(1)
    file_url = f"https://drive.google.com/uc?id={file_id}"
    output_path = f"receipt_{file_id}.jpg"
    gdown.download(file_url, output_path, quiet=True)
    return output_path

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "", 0
    processed = preprocess_image(image)
    result = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
    text = " ".join(result['text'])
    conf_scores = [int(c) for c in result['conf'] if c != '-1']
    avg_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0
    return text, avg_conf

def extract_amount_from_text(text):
    numbers = re.findall(r"\b\d{2,5}(?:[.,]\d{1,2})?\b", text)
    values = [float(n.replace(',', '').replace('EGP', '').strip()) for n in numbers if float(n.replace(',', '').replace('EGP', '').strip()) >= 10]
    values = [v for v in values if v <= 10000]
    return max(values) if values else None

def check_currency(text):
    return "❗ Non-EGP currency" if re.search(r"\$\s*|USD|SAR|€", text, re.IGNORECASE) else ""

def daily_limit_flag(df, row, limit=3000):
    date = row["Date"]
    emp_id = row["Employee ID"]
    day_total = df[(df["Date"] == date) & (df["Employee ID"] == emp_id)]["Amount (EGP)"].sum()
    return "❗ Daily spending limit exceeded" if day_total > limit else ""

def policy_check(row):
    flags = []
    cat = row["Expense Category"].lower().strip()
    amount = row["Amount (EGP)"]
    if cat.startswith("meals") and amount > 100:
        flags.append("Meal exceeds 100 EGP limit")
    if cat.startswith("hotel") and amount > 2500:
        flags.append("Hotel exceeds 2500 EGP limit")
    if cat.startswith("transportation") and amount > 1000:
        flags.append("Transportation exceeds limit")
    return " | ".join(flags)

def check_receipt(row):
    path = download_receipt(row["Upload Receipt"])
    if not path or not os.path.exists(path):
        return "⚠️ Download failed", "", 0
    text, conf = extract_text_from_image(path)
    amount = extract_amount_from_text(text)
    quality = "❗ Low OCR quality" if conf < 70 else ""
    if amount is None:
        return "❓ No amount found", quality, conf
    claimed = row["Amount (EGP)"]
    if abs(claimed - amount) > 10:
        return f"❗ Mismatch: OCR {amount} ≠ Claimed {claimed}", quality, conf
    return "✅ Match", quality, conf

# === Main App ===
st.title("💼 Medical Rep Expense Audit Dashboard")

df = load_data()
receipt_results, currency_flags, daily_spend_flags, policy_flags = [], [], [], []

with st.spinner("🔍 Processing receipts and flags..."):
    for _, row in df.iterrows():
        receipt, _, _ = check_receipt(row)
        receipt_results.append(receipt)
        currency_flags.append(check_currency(receipt))
        daily_spend_flags.append(daily_limit_flag(df, row))
        policy_flags.append(policy_check(row))

    df["Receipt Verification"] = receipt_results
    df["Currency Check"] = currency_flags
    df["Daily Limit Flag"] = daily_spend_flags
    df["Policy Flags"] = policy_flags

    iso = IsolationForest(contamination=0.2, random_state=42)
    df["Anomaly Score"] = iso.fit_predict(df[["Amount (EGP)"]])
    df["Anomaly"] = df["Anomaly Score"].apply(lambda x: "❗ Anomaly" if x == -1 else "")
    df["Review Decision"] = "Pending"

    df["Flag Summary"] = df[[
        "Receipt Verification", "Currency Check", "Daily Limit Flag", "Policy Flags", "Anomaly"
    ]].apply(lambda x: " | ".join([i for i in x if i]), axis=1)

st.success("✅ Audit complete!")
st.dataframe(df[["Employee ID", "Date", "Amount (EGP)", "Flag Summary"]])
st.download_button("Download Results as CSV", data=df.to_csv(index=False), file_name="flagged_expenses.csv")
