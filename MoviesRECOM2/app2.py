import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote
import os
from sklearn.neighbors import NearestNeighbors


# Streamlit Page Configuration
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# TMDB API Key
API_KEY = "b3dd044ee3a2f261651808bfe9d79309"


# Function to Fetch Movie Poster


def poster(movie):
    try:
        movie = movie.rsplit("(", 1)[0].strip()

        if movie.endswith(", The"):
            movie = "The " + movie[:-5]
        elif movie.endswith(", A"):
            movie = "A " + movie[:-3]
        elif movie.endswith(", An"):
            movie = "An " + movie[:-4]

        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={quote(movie)}"
        )

        data = requests.get(url, timeout=5).json()

        if data.get("results"):
            path = data["results"][0].get("poster_path")
            if path:
                return f"https://image.tmdb.org/t/p/w500{path}"

    except Exception:
        pass

    return "https://via.placeholder.com/300x450?text=No+Poster"


# Load Dataset


@st.cache_data
def load_data():
    folder = os.path.dirname(os.path.abspath(__file__))

    ratings = pd.read_csv(
        os.path.join(folder, "file.tsv"),
        sep="\t",
        names=["user", "movie_id", "rating", "time"]
    )

    movies = pd.read_csv(
        os.path.join(folder, "Movie_Id_Titles.csv")
    )

    data = ratings.merge(movies, on="movie_id")

    matrix = data.pivot_table(
        index="user",
        columns="title",
        values="rating"
    )

    return matrix


# Load Ratings Matrix


movie_ratings = load_data()

# Convert to Movie-User Matrix
movie_matrix = movie_ratings.fillna(0).T

# Train KNN Model
model = NearestNeighbors(
    metric="cosine",
    n_neighbors=11
)

model.fit(movie_matrix)


# Streamlit Interface


st.title("🎬 Movie Recommender System")

search = st.text_input("🔍 Enter Movie Name")

matches = (
    sorted(
        [m for m in movie_matrix.index if search.lower() in m.lower()]
    )
    if search else []
)

movie = st.selectbox(
    "Choose Movie",
    matches
) if matches else None


# Recommendation Button


if st.button("Recommend"):

    if not movie:
        st.error("Please select a movie first.")

    else:
        try:
            index = movie_matrix.index.get_loc(movie)

            distances, indices = model.kneighbors(
                movie_matrix.iloc[index].values.reshape(1, -1)
            )

            recommendations = [
                movie_matrix.index[i]
                for i in indices.flatten()[1:]
            ]

            st.subheader("🎬 You Might Also Like")

            cols = st.columns(5)

            for i, name in enumerate(recommendations):
                with cols[i % 5]:
                    st.image(
                        poster(name),
                        use_container_width=True
                    )
                    st.markdown(f"**{name}**")

        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.caption("Developed By: Nangbun Bareh")
