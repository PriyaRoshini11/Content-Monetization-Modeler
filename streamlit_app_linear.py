import streamlit as st
import pandas as pd
import numpy as np
import joblib
from googleapiclient.discovery import build
import isodate
import os

# ==============================
# Load trained pipeline
# ==============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "Linear Regression.pkl")

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="YouTube Ad Revenue Predictor 🎥💰",page_icon="🎬", layout="wide",initial_sidebar_state="expanded")
st.title("🎥 YouTube Ad Revenue Prediction App")

with st.sidebar: 
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/42/YouTube_icon_%282013-2017%29.png", width=100) 
    st.markdown("## 🎥 About the App") 
    st.info("This app predicts YouTube Ad Revenue 💰 using machine learning. \n\n👉 Choose manual entry or paste a YouTube link!")

# Sidebar mode switch
mode = st.sidebar.radio("Choose Input Mode", ["Manual Input", "YouTube Link"])

# ==============================
# Common Prediction Function
# ==============================
def make_prediction(input_dict):
    input_df = pd.DataFrame([input_dict])
    try:
        # Align with model features
        model_features = model.feature_names_in_
        input_df = input_df.reindex(columns=model_features, fill_value=0)

        prediction = model.predict(input_df)[0]
        return prediction
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ==============================
# Manual Input Mode
# ==============================
if mode == "Manual Input":
    st.subheader("🔹 Enter Video Details Manually")

    views = st.number_input("Views", min_value=0, step=1000, value=10000)
    likes = st.number_input("Likes", min_value=0, step=10, value=500)
    comments = st.number_input("Comments", min_value=0, step=5, value=50)
    watch_time_minutes = st.number_input("Watch Time (minutes)", min_value=0.0, step=10.0, value=500.0, format="%.2f")
    video_length_minutes = st.number_input("Video Length (minutes)", min_value=0.1, step=0.1, value=10.0, format="%.2f")
    subscribers = st.number_input("Subscribers", min_value=0, step=100, value=10000)

    category = st.selectbox("Category", ["Entertainment", "Gaming", "Education", "Music", "Tech", "Lifestyle"])
    device = st.selectbox("Device", ["Mobile", "Desktop", "Tablet", "TV"])
    country = st.selectbox("Country", ["US", "IN", "UK", "CA", "DE", "AU"])

    if st.button("🔮 Predict Revenue"):
        engagement_rate = (likes + comments) / views if views > 0 else 0
        avg_watch_time_per_view = watch_time_minutes / views if views > 0 else 0

        input_dict = {
            "views": views,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time_minutes,
            "video_length_minutes": video_length_minutes,
            "subscribers": subscribers,
            "category": category,
            "device": device,
            "country": country,
            "engagement_rate": engagement_rate,
            "avg_watch_time_per_view": avg_watch_time_per_view
        }

        prediction = make_prediction(input_dict)
        if prediction is not None:
            st.success(f"💰 Estimated Ad Revenue: **${prediction:,.2f} USD**")

# ==============================
# YouTube Link Mode
# ==============================
else:
    st.subheader("🔹 Paste a YouTube Link")

    api_key = st.text_input("Enter your YouTube Data API Key", type="password")
    youtube_url = st.text_input("Paste YouTube Video Link")

    def extract_video_id(url: str) -> str:
        if "watch?v=" in url:
            return url.split("watch?v=")[-1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[-1].split("?")[0]
        else:
            return None

    if st.button("Fetch & Predict"):
        if api_key and youtube_url:
            video_id = extract_video_id(youtube_url)
            if not video_id:
                st.error("❌ Invalid YouTube link format.")
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
                        content = item["contentDetails"]

                        st.success(f"✅ Video found: {snippet['title']}")

                        views = int(stats.get("viewCount", 0))
                        likes = int(stats.get("likeCount", 0))
                        comments = int(stats.get("commentCount", 0))
                        duration = content["duration"]
                        video_length_seconds = isodate.parse_duration(duration).total_seconds()
                        video_length_minutes = video_length_seconds / 60
                        mins, secs = divmod(int(video_length_seconds), 60)
                        formatted_duration = f"{mins}:{secs:02d}"

                        # Channel subscribers
                        channel_id = snippet["channelId"]
                        ch_request = youtube.channels().list(part="statistics", id=channel_id)
                        ch_response = ch_request.execute()
                        subscribers = int(ch_response["items"][0]["statistics"].get("subscriberCount", 0))

                        # Approximate watch time
                        watch_time_minutes = views * video_length_minutes

                        # Defaults (can be refined later)
                        category = "Entertainment"
                        device = "Mobile"
                        country = "US"

                        engagement_rate = (likes + comments) / views if views > 0 else 0
                        avg_watch_time_per_view = watch_time_minutes / views if views > 0 else 0

                        input_dict = {
                            "views": views,
                            "likes": likes,
                            "comments": comments,
                            "watch_time_minutes": watch_time_minutes,
                            "video_length_minutes": video_length_minutes,
                            "subscribers": subscribers,
                            "category": category,
                            "device": device,
                            "country": country,
                            "engagement_rate": engagement_rate,
                            "avg_watch_time_per_view": avg_watch_time_per_view
                        }

                        prediction = make_prediction(input_dict)
                        if prediction is not None:
                            st.write(f"📊 Views: {views:,}, 👍 Likes: {likes:,}, 💬 Comments: {comments:,}, 👥 Subsribers: {subscribers:,}, ⏱ Duration: {formatted_duration} (≈ {video_length_minutes:.2f} mins)")
                            st.metric("💰 Predicted Ad Revenue (USD)", value=f"${prediction:,.2f}")

                    else:
                        st.error("❌ Could not fetch video details.")
                except Exception as e:
                    st.error(f"API error: {e}")
        else:
            st.error("Please enter a valid API key and YouTube link.")
            

