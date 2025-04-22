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
    df = pd.read_csv("Spotify.csv", delimiter=";")

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
    # Clean column names again to remove trailing spaces
    filtered_df.columns = filtered_df.columns.str.strip()

    features = ['bpm', 'energy', 'danceability', 'dB', 'liveness', 'valence',
                'duration', 'acousticness', 'speechiness', 'popularity']

    # Check if all features exist
    missing_features = [f for f in features if f not in filtered_df.columns]
    if missing_features:
        st.error(f"Missing features in dataset: {missing_features}")
    else:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(filtered_df[features])

        # Safely get song index
        song_match = filtered_df[filtered_df['title'] == selected_song]
        if song_match.empty:
            st.error("❌ Selected song not found in the filtered dataset.")
        else:
            song_idx = song_match.index[0]

            try:
                similarity_scores = cosine_similarity(
                    [scaled_features[list(filtered_df.index).index(song_idx)]],
                    scaled_features
                )[0]

                filtered_df['similarity'] = similarity_scores
                recommended = filtered_df.sort_values(by='similarity', ascending=False)
                recommended = recommended[recommended['title'] != selected_song]

                st.subheader("🎧 Top 10 Recommended Songs")
                st.dataframe(recommended[['title', 'artist', 'top genre', 'year', 'popularity']].head(10))
            except Exception as e:
                st.error(f"An unexpected error occurred during similarity calculation: {e}")
