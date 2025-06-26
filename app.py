import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import random
import math
import nltk
nltk.download('stopwords')

st.set_page_config(page_icon = ":book", page_title = "Book Recommendation System", layout = "wide")

@st.cache_data(show_spinner= False)
def load_data():
    books = pd.read_csv("datasets/Books.csv")
    ratings = pd.read_csv("datasets/Ratings.csv")
    users = pd.read_csv("datasets/Users.csv")
    return books, ratings, users

with st.spinner("Loading data..."):
    books, ratings, users = load_data()

@st.cache_data(show_spinner= False)
def preprocess_data(books, ratings, users):
    merged_df = pd.merge(users, ratings, on='User-ID')
    merged_df = pd.merge(merged_df, books, on='ISBN')
    merged_df.drop(['Image-URL-S','Image-URL-S','Image-URL-L'], axis=1, inplace=True)
    merged_df['Country'] = merged_df['Location'].astype(str).apply(lambda x: x.split(',')[-1].strip())
    merged_df.drop('Location', axis=1, inplace=True)
    merged_df.columns = [c.replace('-', '_') for c in merged_df.columns]
    merged_df.loc[(merged_df.Age > 100) | (merged_df.Age < 5), 'Age'] = np.nan
    median_age = merged_df['Age'].median()
    std_age = merged_df['Age'].std()
    nulls = merged_df['Age'].isnull().sum()
    random_ages = np.random.randint(median_age - std_age, median_age + std_age, size=nulls)
    merged_df.loc[merged_df['Age'].isnull(), 'Age'] = random_ages
    merged_df['Age'] = merged_df['Age'].astype(int)
    merged_df.dropna(subset=['Book_Author','Publisher'], inplace=True)
    merged_df['Year_Of_Publication'] = pd.to_numeric(merged_df['Year_Of_Publication'], errors='coerce')
    merged_df = merged_df.dropna(subset=['Year_Of_Publication'])
    merged_df['Year_Of_Publication'] = merged_df['Year_Of_Publication'].astype(int)
    return merged_df

with st.spinner("Preprocessing data.."):
    merged_df = preprocess_data(books, ratings, users)

@st.cache_data(show_spinner= False)
def create_content_based_pivot(df):
    df = df[df.Book_Rating != 0]
    user_counts = df.groupby('User_ID')['Book_Rating'].count()
    df = df[df['User_ID'].isin(user_counts[user_counts > 50].index)]
    book_counts = df.groupby('Book_Title')['Book_Rating'].count()
    df = df[df['Book_Title'].isin(book_counts[book_counts > 10].index)]
    pivot = df.pivot_table(index='Book_Title', columns='User_ID', values='Book_Rating').fillna(0)
    return pivot

with st.spinner("Creating pivot table..."):
    pivot_table = create_content_based_pivot(merged_df)

@st.cache_data(show_spinner= False)
def compute_similarity(pivot):
    sim_scores = cosine_similarity(pivot)
    return sim_scores

with st.spinner("Computing Similarity Matrix"):
    similarity_matrix = compute_similarity(pivot_table)

def recommend_content(book_name, n_rec=5):
    try:
        index = np.where(pivot_table.index == book_name)[0][0]
    except IndexError:
        st.error("Book not found in the dataset!")
        return pd.DataFrame()
    sim_scores = list(enumerate(similarity_matrix[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n_rec+1]
    recommended_books = [pivot_table.index[i[0]] for i in sim_scores]
    rec_df = books[books['Book-Title'].isin(recommended_books)]
    return rec_df


@st.cache_data(show_spinner= False)
def create_user_item_matrix(df, sample_size=None):
    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=42)
    try:
        user_item = df.pivot_table(
            index='User_ID',
            columns='Book_Title',
            values='Book_Rating',
            aggfunc='mean',
            fill_value=0
        )
    except Exception as e:
        st.error(f"Error creating user-item matrix: {e}")
        user_item = pd.DataFrame()
    return user_item

with st.spinner("Creating User -Item Matrix..."):
    user_item_matrix = create_user_item_matrix(merged_df, sample_size=10000)

@st.cache_data(show_spinner= False)
def compute_svd(user_item):
    if user_item.empty:
        st.error("User-item matrix is empty. Cannot compute SVD.")
        return pd.DataFrame()
    k = 15  
    U, sigma, Vt = svds(user_item.values.astype(float), k=k)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    preds_df = pd.DataFrame(all_user_predicted_ratings, index=user_item.index, columns=user_item.columns)
    return preds_df

with st.spinner("Computing SVD"):
    svd_preds = compute_svd(user_item_matrix)

def recommend_user_based(user_id, n_rec=5):
    if user_id not in svd_preds.index:
        st.error("User ID not found!")
        return pd.DataFrame()
    user_predictions = svd_preds.loc[user_id].sort_values(ascending=False)
    rated_books = user_item_matrix.loc[user_id]
    rated_books = rated_books[rated_books > 0].index.tolist()
    rec_books = user_predictions[~user_predictions.index.isin(rated_books)].head(n_rec).index.tolist()
    rec_df = books[books['Book-Title'].isin(rec_books)]
    return rec_df

@st.cache_data(show_spinner= False)
def get_overview_df(df):
    book_rating = df.groupby('Book_Title')['Book_Rating'].agg(['mean','count']).reset_index()
    book_rating = book_rating[book_rating['count'] > 200]
    book_rating = book_rating.sort_values(by='mean', ascending=False)
    overview_df = pd.merge(book_rating, books, left_on='Book_Title', right_on='Book-Title')
    return overview_df

with st.spinner("Generating Overview"):
    overview_df = get_overview_df(merged_df)

if 'plots' not in st.session_state:
    with st.spinner("Generating Plots"):    
        st.session_state.plots = {}
        age_counts = merged_df['Age'].value_counts().sort_index()
        fig_age = px.bar(x=age_counts.index, y=age_counts.values, labels={'x': 'Age', 'y': 'Count'},
                        title="Age Distribution of Users")
        st.session_state.plots['age_distribution'] = fig_age

        year_data = merged_df[merged_df['Year_Of_Publication'] > 1800]['Year_Of_Publication']
        fig_year = px.histogram(year_data, nbins=50, title="Year Of Publication Distribution",
                                labels={'value': 'Year Of Publication'})
        st.session_state.plots['year_distribution'] = fig_year

        top_authors = books['Book-Author'].value_counts().head(10).reset_index()
        top_authors.columns = ['Book-Author','Count']
        fig_authors = px.bar(top_authors, x='Count', y='Book-Author', orientation='h',
                            title="Top 10 Authors")
        st.session_state.plots['top_authors'] = fig_authors

        top_publishers = books['Publisher'].value_counts().head(10).reset_index()
        top_publishers.columns = ['Publisher','Count']
        fig_publishers = px.bar(top_publishers, x='Count', y='Publisher', orientation='h',
                                title="Top 10 Publishers")
        st.session_state.plots['top_publishers'] = fig_publishers

        top_countries = merged_df['Country'].value_counts().head(5).reset_index()
        top_countries.columns = ['Country','Count']
        fig_countries = px.pie(top_countries, values='Count', names='Country', title="Top 5 Countries")
        st.session_state.plots['top_countries'] = fig_countries

        book_rating = merged_df.groupby(['Book_Title','Book_Author'])['Book_Rating'].agg(['count','mean']).reset_index()
        top_books = book_rating[book_rating['count']>500].sort_values('mean', ascending=False).head(10)
        fig_top_books = px.bar(top_books, x='mean', y='Book_Title', orientation='h',
                            title="Top 10 Books by Average Rating",
                            labels={'mean':'Average Rating', 'Book_Title':'Book Title'},
                            color='Book_Author')
        st.session_state.plots['top_books'] = fig_top_books

        rating_counts = merged_df['Book_Rating'].value_counts().reset_index()
        rating_counts.columns = ['Book_Rating', 'Count']
        fig_rating = px.bar(rating_counts, x='Book_Rating', y='Count', title="Ratings Distribution")
        st.session_state.plots['rating_distribution'] = fig_rating

st.title("📖 Book Recommendation System")

tabs = st.tabs(["About the Dataset", "Data Visualisations", "Recommendations", "Top Books"])

with tabs[0]:
    st.header("About the Dataset")
    st.subheader("Books Data")
    st.write("**Shape:**", str(books.shape))
    st.dataframe(books.head())
    st.dataframe(books.describe())
    
    st.subheader("Ratings Data")
    st.write("**Shape:**", str(ratings.shape))
    st.dataframe(ratings.head())
    st.dataframe(ratings.describe())
    
    st.subheader("Users Data")
    st.write("**Shape:**", str(users.shape))
    st.dataframe(users.head())
    st.dataframe(users.describe())

with tabs[1]:
    st.header("Data Visualisations")
    for plot in st.session_state.plots.values():
        st.plotly_chart(plot)

with tabs[2]:
    st.header("Recommendations")
    rec_option = st.radio("Choose Recommendation Type", ("Content-Based", "User-Based"))
    if rec_option == "Content-Based":
        st.subheader("Content-Based Recommendation")
        book_name = st.selectbox("Enter Book Name", options = pivot_table.index)
        if st.button("Get Recommendations", key="content"):
            with st.spinner("Getting Recommendations"):
                rec_df = recommend_content(book_name, int(5))
                if not rec_df.empty:
                    st.write("### Recommended Books:")
                    for idx, row in rec_df.iterrows():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(row['Image-URL-S'].replace("THUMBZZZ", "SL1000"), width=200)
                            st.markdown("---")
                        with col2:
                            st.markdown(f"**{row['Book-Title']}** by {row['Book-Author']}")
                            st.write(f"Year: {row['Year-Of-Publication']}, Publisher: {row['Publisher']}")

    else:
        st.subheader("User-Based Recommendation")
        user_ids = user_item_matrix.index.tolist() if not user_item_matrix.empty else []
        selected_user = st.selectbox("Select User ID", user_ids) if user_ids else st.error("No users available.")
        if st.button("Get Recommendations", key="user") and user_ids:
            with st.spinner("Getting Recommendations"):
                rec_df = recommend_user_based(selected_user, 5)
                if not rec_df.empty:
                    st.write("### Recommended Books:")
                    for idx, row in rec_df.iterrows():
                        col1, col2 = st.columns(2)
                        with col1:
                            high_res_url = row['Image-URL-S'].replace("THUMBZZZ", "SL1000")
                            st.image(high_res_url, width=200)
                            st.markdown("---")
                        with col2:
                            st.markdown(f"**{row['Book-Title']}** by {row['Book-Author']}")
                            st.write(f"Year: {row['Year-Of-Publication']}, Publisher: {row['Publisher']}")
                      

with tabs[3]:
    st.header("Overview of Top Books")
    for idx, row in overview_df.iterrows():
        col1, col2 = st.columns(2)
        with col1:
            st.image(row['Image-URL-S'].replace("THUMBZZZ", "SL1000"), width=200)
            st.markdown("---")
        with col2:
            st.markdown(f"**{row['Book-Title']}** by {row['Book-Author']}")
            st.write(f"Year: {row['Year-Of-Publication']}, Publisher: {row['Publisher']}")
