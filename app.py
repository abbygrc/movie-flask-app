# flask
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
import pandas as pd

# FLASK
from flask import Flask, render_template, request, redirect

# SearchMovies.py class
# import SearchMovies as sm


# Start the app, set project root dir as the static folder
app = Flask(__name__)

title = []
service = []
year = []
overview = []
genres = []
images = []
link = []


'''  in html = in python app
movieTitle = title
streamService = service
movieYear = year
movieOverview = overview
movieGenres = genres
moviePoster = images
'''


# Super Hard-Coded
@app.route("/", methods=["GET", "POST"])
def index():
    global counter
    counter = 0
    if request.method == "POST":
        # request.form to access all fields of the form
        print(request.form)
        # FOR SEARCHING THE MOVIES TO PRINT OUT
        searchTitle = request.form.get("movie")
        n = request.form.get("netflix")
        h = request.form.get("hulu")
        d = request.form.get("disney")
        p = request.form.get("prime")

        #search_movies(searchTitle, n, h, d, p)
        results = get_recommendations(searchTitle)
        indices2 = pd.Series(results.index, index=results["Title"])
        resultsFinal = clean_recommendations(indices2)
        resultsFinal.where(
            (movies["Netflix"] == n)
            | (movies["Hulu"] == h)
            | (movies["Disney+"] == d)
            | (movies["Prime Video"] == p)
        ).dropna(how="all")

        if n == "on":
            # print("\n\nNetflix: \n")
            # formatting("Netflix")
            resultsPrint = resultsFinal.where(
                resultsFinal["Netflix"] == 1).dropna()
            for films in resultsPrint.index:
                title.append(resultsPrint["Title"][films])
                service.append("Netflix")
                year.append(resultsPrint["Year"][films])
                overview.append(resultsPrint["Overview"][films])
                genres.append(resultsPrint["Genres"][films])
                images.append(resultsPrint["Image"][films])
                link.append("https://www.netflix.com/")
                counter += 1
                # print("\n" + titleN + "\n\t" + overviewN + "\n\t" + genresN)

        if h == "on":
            # print("\n\nHulu: \n")
            # formatting("Hulu")
            resultsPrint = resultsFinal.where(
                resultsFinal["Hulu"] == 1).dropna()
            for films in resultsPrint.index:
                title.append(resultsPrint["Title"][films])
                service.append("Hulu")
                year.append(resultsPrint["Year"][films])
                overview.append(resultsPrint["Overview"][films])
                genres.append(resultsPrint["Genres"][films])
                images.append(resultsPrint["Image"][films])
                link.append("https://www.hulu.com/")
                counter += 1
                # print("\n" + titleH + "\n\t" + overviewH + "\n\t" + genresH)

        if d == "on":
            # print("\n\nDisney+: \n")
            # formatting("Disney+")
            resultsPrint = resultsFinal.where(
                resultsFinal["Disney+"] == 1).dropna()
            for films in resultsPrint.index:
                title.append(resultsPrint["Title"][films])
                service.append("Disney+")
                year.append(resultsPrint["Year"][films])
                overview.append(resultsPrint["Overview"][films])
                genres.append(resultsPrint["Genres"][films])
                images.append(resultsPrint["Image"][films])
                link.append("https://www.disneyplus.com/")
                counter += 1
                # print("\n" + titleD +"\n\t" + overviewD + "\n\t" + genresD)

        if p == "on":
            # print("\n\nPrime Video: \n")
            # formatting("Prime Video")
            resultsPrint = resultsFinal.where(
                resultsFinal["Prime Video"] == 1).dropna()
            for films in resultsPrint.index:
                title.append(resultsPrint["Title"][films])
                service.append("Amazon Prime")
                year.append(resultsPrint["Year"][films])
                overview.append(resultsPrint["Overview"][films])
                genres.append(resultsPrint["Genres"][films])
                images.append(resultsPrint["Image"][films])
                link.append(
                    "https://www.amazon.com/Prime-Video/b?ie=UTF8&node=2676882011")
                counter += 1
                # print("\n" + titleP + "\n\t" + overviewP + "\n\t" + genresP)

    # Display page
    return render_template("index.html", movieTitle=title, streamService=service, movieYear=year, movieOverview=overview, movieGenres=genres, moviePoster=images, movieLink=link, count=counter)


# SearchMovies.py
# Declare variables
movies = pd.read_csv("moviesNewColumns.csv", low_memory=False)
movies["Year"] = movies["Year"].astype(int)
movies["Year"] = movies["Year"].astype(str)

# Calculating the tfidf and cossim
tfidf = TfidfVectorizer(stop_words="english")

# Apply tfidf and cosine simimlarity to the new tuples
# "Keywords" "Titlekeys" "Advancedkeywords" "CastOverview"

tfidf_matrix = tfidf.fit_transform(movies["Keywords"].astype(str))
cosine_sim = cosine_similarity(tfidf_matrix)

tfidf_matrix2 = tfidf.fit_transform(movies["TitleKeys"].astype(str))
cosine_sim2 = cosine_similarity(tfidf_matrix2)

tfidf_matrix3 = tfidf.fit_transform(movies["AdvancedKeywords"].astype(str))
cosine_sim3 = cosine_similarity(tfidf_matrix3)

tfidf_matrix4 = tfidf.fit_transform(movies["CastOverview"].astype(str))
cosine_sim4 = cosine_similarity(tfidf_matrix4)

# Store movies in variable indices
indices = pd.Series(movies.index, index=movies["Title"])


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):

    # try:
    #     test = indices[title].tolist()
    # except KeyError:
    #     title = (
    #         movies["Title"]
    #         .where(movies["Title"].str.contains(title, regex=False))
    #         .dropna(how="all")
    #         .iloc[0]
    #     )

    title = (
        movies["Title"]
        .where(movies["Title"].str.contains(title, regex=False))
        .dropna(how="all")
        .iloc[0]
    )
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:29]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar movies
    return movies[
        [
            "Title",
            "Year",
            "Overview",
            "Genres",
            "Netflix",
            "Disney+",
            "Hulu",
            "Prime Video",
            "Image",
        ]
    ].iloc[movie_indices]
    # return idx


# Function that takes in movie title as input and outputs most similar movies
def clean_recommendations(indices2, cosine_sim3=cosine_sim3):
    idx = indices2[0]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim3[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[0:9]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return movies[
        [
            "Title",
            "Year",
            "Overview",
            "Genres",
            "Netflix",
            "Disney+",
            "Hulu",
            "Prime Video",
            "Image",
        ]
    ].iloc[movie_indices]
    # return idx
