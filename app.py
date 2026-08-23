"""
Movie Rating Analysis — Full Analytics Website (Streamlit)

Input:  Top_10000_Movies_IMDb.csv in the same folder, with columns:
        ID, Movie Name, Rating, Runtime, Genre, Metascore, Plot,
        Directors (list, first element = director), Stars, Votes, Gross, Link

Run with:  streamlit run app.py

NOTE: this dataset has no release-year column, so year-wise analysis
is not available. Everything else (genre, director, actor, popularity,
box office, correlation, recommendations, prediction) is included.
"""
import ast
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Movie Rating Analysis", layout="wide", page_icon="🎬")

# Make matplotlib charts match the dark theme instead of default white background
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#1C1F26",
    "axes.edgecolor": "#F5F5F5",
    "axes.labelcolor": "#F5F5F5",
    "text.color": "#F5F5F5",
    "xtick.color": "#F5F5F5",
    "ytick.color": "#F5F5F5",
    "grid.color": "#333333",
    "figure.edgecolor": "#0E1117",
    "savefig.facecolor": "#0E1117",
})


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_raw():
    """Raw data, used only for reporting missing values / duplicates."""
    return pd.read_csv("Top_10000_Movies_IMDb.csv")


@st.cache_data
def load_data():
    df = pd.read_csv("Top_10000_Movies_IMDb.csv")

    def get_director(s):
        try:
            lst = ast.literal_eval(s)
            return lst[0] if lst else None
        except Exception:
            return None

    def get_stars(s):
        try:
            lst = ast.literal_eval(s)
            return ", ".join(lst[:4]) if lst else None
        except Exception:
            return None

    df["director"] = df["Directors"].apply(get_director)
    df["cast"] = df["Stars"].apply(get_stars)

    df = df.rename(columns={
        "Movie Name": "title",
        "Rating": "rating",
        "Genre": "genre",
        "Votes": "votes",
        "Gross": "gross",
        "Metascore": "metascore",
        "Plot": "plot",
    })

    df["duration"] = df["Runtime"].astype(str).str.extract(r"(\d+)").astype(float)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce")
    df["gross"] = pd.to_numeric(df["gross"], errors="coerce")
    df["main_genre"] = df["genre"].astype(str).str.split(",").str[0].str.strip()

    return df.dropna(subset=["rating", "main_genre", "votes", "director", "duration", "gross"])


def runtime_bucket(mins):
    if mins < 90:
        return "<90 min"
    elif mins < 120:
        return "90-120 min"
    elif mins < 150:
        return "120-150 min"
    elif mins < 180:
        return "150-180 min"
    else:
        return "180+ min"


raw_df = load_raw()
df = load_data()
df["runtime_bucket"] = df["duration"].apply(runtime_bucket)



@st.cache_resource
def train_all_models(df):
    cat_cols = ["main_genre", "director"]
    encoder_kwargs = dict(handle_unknown="ignore", min_frequency=3, max_categories=200)

    def build_and_train(feature_cols, target_col):
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cats = [c for c in cat_cols if c in feature_cols]
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(**encoder_kwargs), cats),
        ], remainder="passthrough")

        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, n_jobs=-1, random_state=42),
        }
        best = None
        for name, model in candidates.items():
            pipe = Pipeline([("prep", preprocessor), ("model", model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            if best is None or mae < best["MAE"]:
                best = {"pipeline": pipe, "MAE": mae, "R2": r2, "model_name": name}
        return best

    rating_model = build_and_train(["main_genre", "duration", "votes", "director"], "rating")
    gross_model = build_and_train(["main_genre", "duration", "rating", "votes", "director"], "gross")
    votes_model = build_and_train(["main_genre", "duration", "rating", "director"], "votes")

    return {"rating": rating_model, "gross": gross_model, "votes": votes_model}


models = train_all_models(df)

st.title("🎬 Movie Rating Analysis")
st.caption(f"Exploring {len(df):,} movies — full analytics, search, and ML predictions.")

tabs = st.tabs([
    "📊 Dashboard",
    "🔎 Search Movie",
    "🎬 Director Analysis",
    "⭐ Actor Analysis",
    "🔥 Popularity Analysis",
    "💰 Box Office Analysis",
    "🔗 Correlation Analysis",
    "🎯 Recommendations",
    "🔮 Predictions",
])
(tab_dash, tab_movie, tab_director, tab_actor, tab_pop,
 tab_box, tab_corr, tab_rec, tab_pred) = tabs



with tab_dash:
    st.subheader("Overview")

    missing_total = int(raw_df.isna().sum().sum())
    dup_total = int(raw_df.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{len(raw_df):,}")
    c2.metric("Missing Values", f"{missing_total:,}")
    c3.metric("Duplicate Records", f"{dup_total:,}")
    c4.metric("Average Rating", round(df["rating"].mean(), 2))

    top_rated = df.loc[df["rating"].idxmax()]
    top_voted = df.loc[df["votes"].idxmax()]
    top_gross = df.loc[df["gross"].idxmax()]

    c5, c6, c7 = st.columns(3)
    c5.metric("Highest Rated Movie", top_rated["title"], f"{top_rated['rating']}")
    c6.metric("Most Voted Movie", top_voted["title"], f"{int(top_voted['votes']):,} votes")
    c7.metric("Highest Grossing Movie", top_gross["title"], f"${int(top_gross['gross']):,}")

    st.markdown("**Top 10 Directors (by number of movies)**")
    top10_directors = df["director"].value_counts().head(10)
    st.dataframe(top10_directors.rename_axis("Director").reset_index(name="Movies"))

    st.divider()

    # ---------- Genre Insights ----------
    with st.expander("🎭 Genre Insights", expanded=True):
        genre_avg_rating = df.groupby("main_genre")["rating"].mean().sort_values(ascending=False)
        genre_votes = df.groupby("main_genre")["votes"].sum().sort_values(ascending=False)
        genre_count = df["main_genre"].value_counts()

        gc1, gc2 = st.columns(2)
        gc1.metric("Highest Rated Genre", genre_avg_rating.index[0], round(genre_avg_rating.iloc[0], 2))
        gc2.metric("Genre with Most Votes", genre_votes.index[0], f"{int(genre_votes.iloc[0]):,}")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Average Rating by Genre")
            fig, ax = plt.subplots()
            genre_avg_rating.plot(kind="bar", ax=ax, color="skyblue")
            ax.set_ylabel("Average Rating")
            st.pyplot(fig)
        with c2:
            st.markdown("Total Votes by Genre")
            fig, ax = plt.subplots()
            genre_votes.plot(kind="bar", ax=ax, color="orange")
            ax.set_ylabel("Total Votes")
            st.pyplot(fig)

        st.markdown("Genre Frequency (Number of Movies)")
        fig, ax = plt.subplots()
        genre_count.plot(kind="bar", ax=ax, color="teal")
        ax.set_ylabel("Number of Movies")
        st.pyplot(fig)

    # ---------- Director Insights ----------
    with st.expander("🎬 Director Insights"):
        director_counts = df["director"].value_counts().head(10)
        # require at least 3 movies so "highest rated director" isn't a fluke of 1 film
        eligible_directors = df["director"].value_counts()
        eligible_directors = eligible_directors[eligible_directors >= 3].index
        director_avg_rating = (
            df[df["director"].isin(eligible_directors)]
            .groupby("director")["rating"].mean().sort_values(ascending=False).head(10)
        )
        director_gross = df.groupby("director")["gross"].sum().sort_values(ascending=False).head(10)
        director_votes = df.groupby("director")["votes"].sum().sort_values(ascending=False).head(10)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Directors with Most Movies")
            fig, ax = plt.subplots()
            director_counts.plot(kind="bar", ax=ax, color="coral")
            ax.set_ylabel("Number of Movies")
            st.pyplot(fig)
        with c2:
            st.markdown("Highest Rated Directors (min. 3 movies)")
            fig, ax = plt.subplots()
            director_avg_rating.plot(kind="bar", ax=ax, color="mediumseagreen")
            ax.set_ylabel("Average Rating")
            st.pyplot(fig)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("Directors with Highest Gross")
            fig, ax = plt.subplots()
            director_gross.plot(kind="bar", ax=ax, color="gold")
            ax.set_ylabel("Total Gross ($)")
            st.pyplot(fig)
        with c4:
            st.markdown("Directors with Most Votes")
            fig, ax = plt.subplots()
            director_votes.plot(kind="bar", ax=ax, color="slateblue")
            ax.set_ylabel("Total Votes")
            st.pyplot(fig)

    # ---------- Relationships ----------
    with st.expander("📈 Rating Relationships"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("Average Rating by Runtime")
            order = ["<90 min", "90-120 min", "120-150 min", "150-180 min", "180+ min"]
            runtime_avg = df.groupby("runtime_bucket")["rating"].mean().reindex(order)
            fig, ax = plt.subplots()
            runtime_avg.plot(kind="bar", ax=ax, color="plum")
            ax.set_ylabel("Average Rating")
            st.pyplot(fig)
        with c2:
            st.markdown("Rating vs Gross")
            fig, ax = plt.subplots()
            ax.scatter(df["gross"], df["rating"], alpha=0.3, color="green")
            ax.set_xlabel("Gross ($)")
            ax.set_ylabel("Rating")
            st.pyplot(fig)

        st.markdown("Rating vs Votes")
        fig, ax = plt.subplots()
        ax.scatter(df["votes"], df["rating"], alpha=0.3)
        ax.set_xlabel("Votes")
        ax.set_ylabel("Rating")
        st.pyplot(fig)


# ============================================================
# SEARCH MOVIE
# ============================================================
with tab_movie:
    st.subheader("Search for a movie")
    movie_titles = sorted(df["title"].unique())
    selected_title = st.selectbox(
        "Start typing a movie name", options=movie_titles, index=None,
        placeholder="e.g. The Shawshank Redemption", key="movie_search",
    )
    if selected_title:
        row = df[df["title"] == selected_title].iloc[0]
        st.markdown(f"### {row['title']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rating", row["rating"])
        c2.metric("Votes", f"{int(row['votes']):,}")
        c3.metric("Duration", f"{int(row['duration'])} min")
        c4.metric("Gross", f"${int(row['gross']):,}")
        st.write(f"**Genre:** {row['genre']}  |  **Director:** {row['director']}")
        st.write(f"**Cast:** {row['cast']}")
        st.write(row["plot"])


# ============================================================
# DIRECTOR ANALYSIS
# ============================================================
with tab_director:
    st.subheader("Director Analysis")
    st.caption("Search by movie name to jump straight to its director's full profile, or search a director directly.")

    mode = st.radio("Search by", ["Movie name", "Director name"], horizontal=True, key="director_mode")

    selected_director = None
    if mode == "Movie name":
        movie_titles = sorted(df["title"].unique())
        picked_movie = st.selectbox(
            "Start typing a movie name", options=movie_titles, index=None,
            placeholder="e.g. Inception", key="director_via_movie",
        )
        if picked_movie:
            row = df[df["title"] == picked_movie].iloc[0]
            st.markdown(f"**{row['title']}** — Rating: {row['rating']} | Votes: {int(row['votes']):,} | Gross: ${int(row['gross']):,}")
            selected_director = row["director"]
    else:
        director_names = sorted(df["director"].unique())
        selected_director = st.selectbox(
            "Start typing a director name", options=director_names, index=None,
            placeholder="e.g. Christopher Nolan", key="director_direct",
        )

    if selected_director:
        dmatches = df[df["director"] == selected_director]
        st.markdown(f"### {selected_director}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Movies", len(dmatches))
        c2.metric("Average Rating", round(dmatches["rating"].mean(), 2))
        c3.metric("Total Votes", f"{int(dmatches['votes'].sum()):,}")
        c4.metric("Total Gross", f"${int(dmatches['gross'].sum()):,}")

        st.markdown("**All movies by this director**")
        st.dataframe(
            dmatches.sort_values("rating", ascending=False)[
                ["title", "main_genre", "rating", "votes", "gross", "duration"]
            ].reset_index(drop=True)
        )

        fig, ax = plt.subplots()
        top_movies = dmatches.sort_values("rating", ascending=False).head(20)
        ax.bar(top_movies["title"], top_movies["rating"], color="coral")
        ax.set_ylabel("Rating")
        ax.set_xticklabels(top_movies["title"], rotation=75, ha="right")
        st.pyplot(fig)


# ============================================================
# ACTOR ANALYSIS
# ============================================================
with tab_actor:
    st.subheader("Actor Analysis")
    st.caption("Search by movie name to see its cast, or search an actor directly.")

    all_actors = sorted(set(
        name.strip()
        for cast_str in df["cast"].dropna()
        for name in cast_str.split(",")
        if name.strip()
    ))

    mode = st.radio("Search by", ["Movie name", "Actor name"], horizontal=True, key="actor_mode")

    selected_actor = None
    if mode == "Movie name":
        movie_titles = sorted(df["title"].unique())
        picked_movie = st.selectbox(
            "Start typing a movie name", options=movie_titles, index=None,
            placeholder="e.g. The Godfather", key="actor_via_movie",
        )
        if picked_movie:
            row = df[df["title"] == picked_movie].iloc[0]
            st.markdown(f"**Cast of {row['title']}:** {row['cast']}")
            cast_list = [c.strip() for c in str(row["cast"]).split(",") if c.strip()]
            if cast_list:
                selected_actor = st.selectbox("Pick an actor from this cast", cast_list, key="actor_pick")
    else:
        selected_actor = st.selectbox(
            "Start typing an actor name", options=all_actors, index=None,
            placeholder="e.g. Tim Robbins", key="actor_direct",
        )

    if selected_actor:
        amatches = df[df["cast"].str.contains(selected_actor, case=False, na=False)]
        st.markdown(f"### {selected_actor}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Movies", len(amatches))
        c2.metric("Average Rating", round(amatches["rating"].mean(), 2))
        c3.metric("Total Votes", f"{int(amatches['votes'].sum()):,}")
        c4.metric("Total Gross", f"${int(amatches['gross'].sum()):,}")

        st.markdown("**Movies featuring this actor**")
        st.dataframe(
            amatches.sort_values("rating", ascending=False)[
                ["title", "main_genre", "rating", "votes", "gross", "director", "duration"]
            ].reset_index(drop=True)
        )

        fig, ax = plt.subplots()
        top_movies = amatches.sort_values("rating", ascending=False).head(20)
        ax.bar(top_movies["title"], top_movies["rating"], color="mediumpurple")
        ax.set_ylabel("Rating")
        ax.set_xticklabels(top_movies["title"], rotation=75, ha="right")
        st.pyplot(fig)


# ============================================================
# POPULARITY ANALYSIS (votes-based)
# ============================================================
with tab_pop:
    st.subheader("Popularity Analysis (based on votes)")

    st.markdown("**Top 10 Most Voted Movies**")
    top_voted = df.sort_values("votes", ascending=False).head(10)
    st.dataframe(top_voted[["title", "main_genre", "rating", "votes", "director"]].reset_index(drop=True))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Votes Distribution")
        fig, ax = plt.subplots()
        ax.hist(df["votes"], bins=40, color="steelblue")
        ax.set_xlabel("Votes")
        ax.set_ylabel("Number of Movies")
        st.pyplot(fig)
    with c2:
        st.markdown("Popularity (Votes) vs Rating")
        fig, ax = plt.subplots()
        ax.scatter(df["votes"], df["rating"], alpha=0.3, color="indigo")
        ax.set_xlabel("Votes")
        ax.set_ylabel("Rating")
        st.pyplot(fig)


# ============================================================
# BOX OFFICE ANALYSIS
# ============================================================
with tab_box:
    st.subheader("Box Office Analysis (based on gross)")

    st.markdown("**Top 10 Highest Grossing Movies**")
    top_gross = df.sort_values("gross", ascending=False).head(10)
    st.dataframe(top_gross[["title", "main_genre", "rating", "gross", "director"]].reset_index(drop=True))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Average Gross by Genre")
        genre_gross = df.groupby("main_genre")["gross"].mean().sort_values(ascending=False)
        fig, ax = plt.subplots()
        genre_gross.plot(kind="bar", ax=ax, color="darkgoldenrod")
        ax.set_ylabel("Average Gross ($)")
        st.pyplot(fig)
    with c2:
        st.markdown("Gross vs Runtime")
        fig, ax = plt.subplots()
        ax.scatter(df["duration"], df["gross"], alpha=0.3, color="seagreen")
        ax.set_xlabel("Duration (min)")
        ax.set_ylabel("Gross ($)")
        st.pyplot(fig)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================
with tab_corr:
    st.subheader("Correlation Analysis")
    numeric_cols = ["rating", "votes", "gross", "duration", "metascore"]
    corr_df = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
    ax.set_yticklabels(numeric_cols)
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            ax.text(j, i, f"{corr_df.iloc[i, j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)
    st.caption("Correlation between rating, votes, gross, duration, and metascore. Values close to 1 or -1 indicate a strong relationship.")


# ============================================================
# RECOMMENDATIONS
# ============================================================
with tab_rec:
    st.subheader("Movie Recommendations")
    fav_genre = st.selectbox("Pick your favorite genre", sorted(df["main_genre"].unique()))
    top10 = (
        df[df["main_genre"] == fav_genre]
        .sort_values("rating", ascending=False)
        .head(10)[["title", "rating", "votes", "director", "duration"]]
    )
    st.markdown(f"**Top 10 {fav_genre} Movies**")
    st.dataframe(top10.reset_index(drop=True))


# ============================================================
# PREDICTIONS
# ============================================================
with tab_pred:
    st.subheader("Predict IMDb Rating, Gross, or Votes")
    target = st.radio("What do you want to predict?", ["IMDb Rating", "Gross Collection", "Votes"], horizontal=True)

    genre_in = st.selectbox("Genre", sorted(df["main_genre"].unique()), key="pred_genre")
    director_in = st.selectbox("Director", sorted(df["director"].unique()), key="pred_director")
    duration_in = st.slider("Duration (min)", 60, 240, 120, key="pred_duration")

    if target == "IMDb Rating":
        votes_in = st.number_input("Votes", min_value=0, value=100000, step=1000, key="pred_votes_for_rating")
        info = models["rating"]
        st.caption(f"Model: {info['model_name']} | MAE: {info['MAE']:.2f} | R²: {info['R2']:.2f}")
        if st.button("Predict Rating"):
            sample = pd.DataFrame([{"main_genre": genre_in, "duration": duration_in, "votes": votes_in, "director": director_in}])
            pred = info["pipeline"].predict(sample)[0]
            st.success(f"Predicted IMDb Rating: **{pred:.2f} / 10**")

    elif target == "Gross Collection":
        rating_in = st.slider("IMDb Rating", 1.0, 10.0, 7.0, key="pred_rating_for_gross")
        votes_in = st.number_input("Votes", min_value=0, value=100000, step=1000, key="pred_votes_for_gross")
        info = models["gross"]
        st.caption(f"Model: {info['model_name']} | MAE: ${info['MAE']:,.0f} | R²: {info['R2']:.2f}")
        if st.button("Predict Gross"):
            sample = pd.DataFrame([{"main_genre": genre_in, "duration": duration_in, "rating": rating_in, "votes": votes_in, "director": director_in}])
            pred = info["pipeline"].predict(sample)[0]
            st.success(f"Predicted Gross Collection: **${pred:,.0f}**")

    else:
        rating_in = st.slider("IMDb Rating", 1.0, 10.0, 7.0, key="pred_rating_for_votes")
        info = models["votes"]
        st.caption(f"Model: {info['model_name']} | MAE: {info['MAE']:,.0f} | R²: {info['R2']:.2f}")
        if st.button("Predict Votes"):
            sample = pd.DataFrame([{"main_genre": genre_in, "duration": duration_in, "rating": rating_in, "director": director_in}])
            pred = info["pipeline"].predict(sample)[0]
            st.success(f"Predicted Votes: **{pred:,.0f}**")
