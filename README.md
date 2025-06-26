# Book Recommendation System

A Streamlit-based web application that provides two types of book recommendations:

1. **Content-Based Filtering**
2. **User-Based Collaborative Filtering (SVD)**

The project also includes data visualizations for better insights into the dataset.

## Project Overview

This **Book Recommendation System** uses two main approaches:

1. **Content-Based Filtering**:  
   Recommends books based on similarity in user ratings and content features, computed via a cosine similarity matrix.

2. **User-Based Collaborative Filtering (SVD)**:  
   Uses Singular Value Decomposition (SVD) on a user–item matrix to predict and recommend books that a user has not yet rated.

The web interface is built with **Streamlit**, making it easy to interact with the recommendation engine, explore the data, and visualize key insights.

---

## Features

1. **Data Preprocessing**

   - Cleans and merges book, ratings, and user datasets.
   - Handles missing values, out-of-range ages, and invalid publication years.
   - Extracts country information from user location.

2. **Content-Based Recommendations**

   - Uses a pivot table of book titles vs. users.
   - Computes a cosine similarity matrix to find similar books.

3. **User-Based Recommendations (Collaborative Filtering)**

   - Creates a user–item matrix.
   - Applies SVD to predict missing ratings.
   - Recommends books for a given user ID based on highest predicted ratings.

4. **Data Visualization**

   - **Age Distribution**
   - **Year of Publication Distribution**
   - **Top 10 Authors**
   - **Top 10 Publishers**
   - **Top 5 Countries**
   - **Top 10 Books by Average Rating**
   - **Ratings Distribution**

5. **Top Books Overview**
   - Displays highly rated books based on average rating and minimum number of ratings.

---

## Project Structure

A typical layout of the repository might look like this:

```
BOOK RECOMMENDATION SYSTEM/
├── datasets/
│   ├── Books.csv
│   ├── Ratings.csv
│   └── Users.csv
├── app.py
├── requirements.txt
├── README.md
```

- **datasets/**: Contains the CSV files used for books, users, and ratings data.
- **app.py**: The main Streamlit application code.
- **requirements.txt**: Lists all the Python dependencies needed to run the application.
- **README.md**: Documentation (this file).

---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Jnan-py/book-recommendation-system.git
   cd book-recommendation-system
   ```

2. **Install dependencies** (preferably within a virtual environment):

   ```bash
   pip install -r requirements.txt
   ```

   _Note:_ If you prefer `conda`, you can create a new environment and install packages there.

3. **Download NLTK stopwords** (if not downloaded already, though the code includes a check):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

## Usage

1. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to:

   ```
   http://localhost:8501
   ```

3. **Explore**:
   - **About the Dataset**: View the shape, head, and descriptive statistics of the Books, Ratings, and Users datasets.
   - **Data Visualisations**: Explore interactive charts and graphs.
   - **Recommendations**:
     - **Content-Based**: Select a book from the dropdown to get similar book recommendations.
     - **User-Based**: Select a user ID to get personalized recommendations.
   - **Top Books**: View the highest-rated books and their details.

---

## Datasets

This project relies on three CSV files:

- **Books.csv**
- **Ratings.csv**
- **Users.csv**

## Technologies Used

- **[Python 3.x](https://www.python.org/)**
- **[Streamlit](https://streamlit.io/)**
- **[Pandas](https://pandas.pydata.org/)**
- **[NumPy](https://numpy.org/)**
- **[Plotly](https://plotly.com/python/)**
- **[Scikit-learn](https://scikit-learn.org/)**
- **[SciPy](https://scipy.org/)**
- **[NLTK](https://www.nltk.org/)**
