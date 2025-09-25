import streamlit as st
import pandas as pd
import numpy as np
import joblib
from googleapiclient.discovery import build
import isodate

# ---------------- Load saved artifacts ----------------
MODEL_PATH = r"C:\Users\Projects\VSPython\Project3\best_gb_model.pkl"
ENCODER_PATH = r"C:\Users\Projects\VSPython\Project3\ordinal_encoder.pkl"
SCALER_PATH = r"C:\Users\Projects\VSPython\Project3\scaler.pkl"
FEATURES_PATH = r"C:\Users\Projects\VSPython\Project3\feature_names.pkl"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

st.set_page_config(page_title="YouTube Ad Revenue Predictor 🎥💰",page_icon="🎬",layout="wide",initial_sidebar_state="expanded")
st.title("🎥 YouTube Ad Revenue Prediction App")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png", width=100)
    st.markdown("## 🎥 About the App")
    st.info("This app predicts YouTube Ad Revenue 💰 using machine learning. \n\n👉 Choose manual entry or paste a YouTube link!")

# ---------------- Helper function ----------------
def build_input(views, likes, comments, watch_time_minutes, video_length_minutes,
                subscribers, category, device, country,
                upload_month, upload_day, upload_year):

    # Encode categories
    cat_df = pd.DataFrame([[category, device, country]], columns=["category", "device", "country"])
    cat_encoded = encoder.transform(cat_df)[0]
    category_enc, device_enc, country_enc = cat_encoded

    # Feature engineering
    engagement = (likes + comments) / views if views > 0 else 0
    watch_per_view = watch_time_minutes / views if views > 0 else 0
    views_per_min = views / video_length_minutes if video_length_minutes > 0 else 0

    # Build row in the SAME ORDER as training
    input_data = pd.DataFrame([[
        views, likes, comments, watch_time_minutes, video_length_minutes,
        subscribers, category_enc, device_enc, country_enc,
        upload_month, upload_day, upload_year,
        engagement, watch_per_view, views_per_min
    ]], columns=feature_names)

    # Scale
    input_scaled = scaler.transform(input_data)

    return input_scaled

# ---------------- Sidebar mode switch ----------------
mode = st.sidebar.radio("Choose Input Mode", ["Manual Input", "YouTube Link"])

# ---------------- Manual Input ----------------
if mode == "Manual Input":
    st.subheader("🔹 Enter Video Details Manually")

    # Integer inputs
    views = st.number_input("Views", min_value=0, step=1000)
    likes = st.number_input("Likes", min_value=0, step=10)
    comments = st.number_input("Comments", min_value=0, step=5)
    
    # Float inputs
    watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0.0, step=10.0, format="%.2f")
    video_length_minutes = st.number_input("Video Length (minutes)", min_value=0.1, step=0.1, format="%.2f")

    # Integer input
    subscribers = st.number_input("Subscribers", min_value=0, step=100)

    # Categories directly from encoder
    category = st.selectbox("Category", options=encoder.categories_[0])
    device   = st.selectbox("Device", options=encoder.categories_[1])
    country  = st.selectbox("Country", options=encoder.categories_[2])

    # Date parts
    upload_month = st.slider("Upload Month", 1, 12, 6)
    upload_day = st.slider("Upload Day", 1, 31, 15)
    upload_year = st.slider("Upload Year", 2015, 2025, 2023)

    if st.button("Predict Revenue 💰"):
        input_scaled = build_input(
            views, likes, comments, watch_time_minutes, video_length_minutes,
            subscribers, category, device, country,
            upload_month, upload_day, upload_year
        )
        prediction = model.predict(input_scaled)[0]
        st.metric(label="💰 Predicted Ad Revenue (USD)", value=f"${prediction:,.2f}")

# ---------------- YouTube Link ----------------
else:
    st.subheader("🔹 Paste a YouTube Link")

    api_key = st.text_input("Enter your YouTube Data API Key", type="password")
    youtube_url = st.text_input("Paste YouTube Video Link")

    if st.button("Fetch & Predict"):
        if api_key and youtube_url:

            # 🟢 Handle both "watch?v=" and "youtu.be" links
            def extract_video_id(url: str) -> str:
                if "watch?v=" in url:
                    return url.split("watch?v=")[-1].split("&")[0]
                elif "youtu.be/" in url:
                    return url.split("youtu.be/")[-1].split("?")[0]
                else:
                    return None

            video_id = extract_video_id(youtube_url)

            if not video_id:
                st.error("❌ Invalid YouTube link format. Try full or short link.")
            else:
                try:
                    youtube = build("youtube", "v3", developerKey=api_key)
                    request = youtube.videos().list(
                        part="snippet,statistics,contentDetails",
                        id=video_id
                    )
                    response = request.execute()

                    if "items" in response and len(response["items"]) > 0:
                        item = response["items"][0]
                        stats = item["statistics"]
                        snippet = item["snippet"]

                        st.success(f"✅ Video found: {snippet['title']}")

                        # Extract stats
                        views = int(stats.get("viewCount", 0))
                        likes = int(stats.get("likeCount", 0))
                        comments = int(stats.get("commentCount", 0))
                        duration = item["contentDetails"]["duration"]
                        duration_minutes = isodate.parse_duration(duration).total_seconds() / 60

                        upload_date = pd.to_datetime(snippet["publishedAt"])
                        upload_month, upload_day, upload_year = upload_date.month, upload_date.day, upload_date.year

                        # Fetch channel subscribers
                        channel_id = snippet["channelId"]
                        ch_request = youtube.channels().list(part="statistics", id=channel_id)
                        ch_response = ch_request.execute()
                        subscribers = int(ch_response["items"][0]["statistics"].get("subscriberCount", 0))

                        # Estimate watch_time
                        watch_time_minutes = duration_minutes * views

                        # Defaults (improve later by mapping)
                        category, device, country = "Music", "Mobile", "US"

                        # Build input + predict
                        input_scaled = build_input(
                            views, likes, comments, watch_time_minutes, duration_minutes,
                            subscribers, category, device, country,
                            upload_month, upload_day, upload_year
                        )
                        prediction = model.predict(input_scaled)[0]
                        st.write(f"📊 Views: {views}, 👍 Likes: {likes}, 💬 Comments: {comments}, 👥 Subs: {subscribers}, ⏱ Duration: {round(duration_minutes,2)} mins")
                        st.metric(label="💰 Predicted Ad Revenue (USD)", value=f"${prediction:,.2f}")

                    else:
                        st.error("❌ Could not fetch video details. Double-check link or API key.")
                except Exception as e:
                    st.error(f"API error: {e}")
        else:
            st.error("Please enter a valid API key and YouTube link.")