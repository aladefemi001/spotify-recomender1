# 👉 Paste your entire Streamlit app code between the triple quotes below

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.title("🎵 Spotify Song Recommender")

# Upload dataset
uploaded_file = st.file_uploader("Upload your Spotify dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("Spotify.csv",delimiter=";")

# Clean column names
df.columns = df.columns.str.strip()

# Optional filters
years = sorted(df['year'].unique())
genres = sorted(df['top genre'].dropna().unique())

col1, col2 = st.columns(2)
year_filter = col1.selectbox("Filter by Year (optional)", options=["All"] + list(years))
genre_filter = col2.selectbox("Filter by Genre (optional)", options=["All"] + list(genres))

filtered_df = df.copy()
if year_filter != "All":
    filtered_df = filtered_df[filtered_df["year"] == year_filter]
if genre_filter != "All":
    filtered_df = filtered_df[filtered_df["top genre"] == genre_filter]

song_titles = filtered_df["title"].tolist()
selected_song = st.selectbox("Select a song you like", options=song_titles)

if st.button("Recommend"):
    features = ['bpm', 'energy', 'danceability ', 'dB', 'liveness', 'valence',
                'duration', 'acousticness', 'speechiness ', 'popularity']

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(filtered_df[features])

    song_idx = filtered_df[filtered_df['title'] == selected_song].index[0]
    similarity_scores = cosine_similarity([scaled_features[song_idx]], scaled_features)[0]

    filtered_df['similarity'] = similarity_scores
    recommended = filtered_df.sort_values(by='similarity', ascending=False)
    recommended = recommended[recommended['title'] != selected_song]

    st.subheader("🎧 Top 10 Recommended Songs")
    st.dataframe(recommended[['title', 'artist', 'top genre', 'year', 'popularity']].head(10))
