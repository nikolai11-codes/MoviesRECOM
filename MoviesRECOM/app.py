import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote
import os

st.set_page_config(                      #Configure/set up the streamlit app
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


API_KEY = st.secrets["TMDB_API_KEY"]     #Reads the key securely from streamlit secrets


def clean_title(title):                      #clean movie title before searching on tmdb
    title = title.rsplit("(", 1)[0].strip()

    if title.endswith(", The"):
        title = "The " + title[:-5]
    elif title.endswith(", A"):
        title = "A " + title[:-3]
    elif title.endswith(", An"):
        title = "An " + title[:-4]

    return title


def poster(movie):                           #fetch movie posters from tmdb api
    try:
        movie = clean_title(movie)

        url = (
            f"https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}&query={quote(movie)}"
        )

        r = requests.get(url, timeout=5).json()

        if r.get("results") and r["results"][0].get("poster_path"):
            return f"https://image.tmdb.org/t/p/w500{r['results'][0]['poster_path']}"

        return None

    except Exception:
        return None


@st.cache_data                                                #load and process the movie DB
def load():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    ratings = pd.read_csv(
        os.path.join(BASE_DIR, "file.tsv"),
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"]
    )

    movies = pd.read_csv(
        os.path.join(BASE_DIR, "Movie_Id_Titles.csv")
    )

    data = ratings.merge(movies, on="item_id")                 #merge ratings with movie titles

    moviemat = data.pivot_table(                               #creates a table of users and their movie ratings
        index="user_id",
        columns="title",
        values="rating"
    )

    stats = data.groupby("title")["rating"].agg(["mean", "count"])          #calculates average rating N rating count for each movie

    return moviemat, stats


moviemat, stats = load()

st.title("🎬 Movie Recommender System")

movie = st.text_input("🔍 Enter movie name")

choices = (
    sorted([m for m in moviemat.columns if movie.lower() in m.lower()])
    if movie else []
)

selected = st.selectbox("Choose movie", choices) if choices else None

if st.button("Recommend"):

    if not selected:
        st.error("Please select a movie first.")

    else:
        corr = moviemat.corrwith(moviemat[selected]).dropna()            #calculates similarity using collaborative filtering
        corr = corr.to_frame("Correlation")
        corr = corr.join(stats["count"])
        corr = corr[corr["count"] > 100]
        corr = corr.sort_values("Correlation", ascending=False)

        recommendations = corr.index[1:11]

        st.subheader("🎬 You Might Also Like")

        cols = st.columns(5)

        for i, name in enumerate(recommendations):                      #Display recommended movies with posters
            with cols[i % 5]:
                img = poster(name)

                st.image(
                    img if img else "https://via.placeholder.com/300x450?text=No+Poster",
                    use_container_width=True
                )

                st.markdown(f"**{name}**")

st.caption("Developed By: Nangbun Bareh")
