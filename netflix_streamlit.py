#!/usr/bin/env python
# coding: utf-8

# In[77]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


#Set the Page Layout
st.set_page_config(page_title='Netflix Dashboard', layout = 'wide')

# Main Header
st.title("Netflix Exploratory Data Dashboard")

st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    h1, h2, h3 {
        color: #C51162;
    }
    </style>
""", unsafe_allow_html=True)


#SideBar
st.sidebar.title("Netflix Dashboard")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Overview", 
    "Content Insights", 
    "Duration Analysis", 
    "Text Analysis", 
    "Trends", 
    "Sentiment", 
    "Recommender"
])

#Load Data
@st.cache_data
def load_data():
    with st.spinner("Loading Netflix dataset..."):
         df = pd.read_csv('netflix_titles.csv')
    return df
df = load_data()

#Sidebar Filters
st.sidebar.header('Filter Netflix Titles')

#Filter by Type
type_filter = st.sidebar.multiselect("Select Type", options=df['type'].dropna().unique(), default = df['type'].dropna().unique(),
                                     help="Select one or more type to filter Netflix titles")

#Filter by Country
country_filter = st.sidebar.multiselect("Select Country", options=df['country'].dropna().unique(), default = ['United States'],
                                       help="Select one or more countries to filter Netflix titles")

#Filter by Release Year
year_filter = st.sidebar.slider("Select Release Year", int(df['release_year'].min()), int(df['release_year'].max()), (2010,2020),
                               help="Use this slider to filter Netflix titles released between selected years")

#Apply Filters
df_filtered = df[(df['type'].isin(type_filter)) & (df['country'].isin(country_filter)) & (df['release_year'] >= year_filter[0]) & (df['release_year'] <= year_filter[1])]

#Show Raw Data
if st.checkbox("Show raw data"):
    st.subheader("RAW DATA")
    st.dataframe(df_filtered)

with tab1:
    st.subheader("Overview")
    st.caption("A high-level snapshot showing Netflix content is distributed by type (Movie or TV Show) and added each year.")

    with st.expander("See Distribution of Content Type"):
         st.subheader("Distribution of Content Type")
         #Count of each type
         type_counts = df_filtered['type'].value_counts()

         #Plot
         sns.set(style='whitegrid')
         plt.figure(figsize=(12,6))
         sns.barplot(x=type_counts.index,y=type_counts.values,hue= type_counts.index, palette = 'pastel', legend = False)
         plt.title('Distribution of Content Type of Netlfix')
         plt.xlabel('Year')
         plt.ylabel('Count')

         st.pyplot(plt.gcf())   #Render the Plot in Streamlit
         plt.clf()    #Clear after Rendering

         st.markdown("---")

    with st.expander("See Number of Movies and TV Shows Added Each Year"):
        st.subheader("Number of Movies and TV Shows Added Each Year")
        #Make a copy to avoid modifying original dataframe
        df_year = df_filtered.copy()

        # Convert 'date_added' to datetime if not already done (important)
        df_year['date_added'] = pd.to_datetime(df_year['date_added'], errors='coerce')

        #Extract year from date_added column
        df_year['year_added'] = df_year['date_added'].dt.year

        #Drop rows where year added is NaN
        df_year = df_year.dropna(subset=['year_added'])

        #Group by year and type and then count
        content_per_year = df_year.groupby(['year_added','type']).size().reset_index(name='count')

        #Plot
        plt.figure(figsize=(12,6))
        sns.lineplot(data=content_per_year,x='year_added',y='count',hue='type')
        plt.title('Number of Movies and TV Shows Added Each Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        plt.grid(True)


        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("---")

with tab2:
    st.subheader("Content Insights")
    st.caption("Explore how content is distributed across genres and directors.")

    with st.expander("See Top 10 Directors with Most Content"):
        st.subheader("Top 10 Directors with Most Content")
        directors = df_filtered[df_filtered['director'].notna() & (df_filtered['director'] != 'Not Available')]
        top_directors = directors['director'].value_counts().head(10).reset_index()
        top_directors.columns = ['Director', 'Count']

        plt.figure(figsize=(10,5))
        sns.barplot(data = top_directors, x = 'Count', y = 'Director', palette = 'Set2', legend=False)
        plt.title('Top 10 Directors with Most Content on Netflix')
        plt.xlabel('Number of titles')
        plt.ylabel('Director')
        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("---")
    
    with st.expander("See Top 10 Genres by Number of Titles"):
        st.subheader("Top 10 Genres by Number of Titles")

        df_genres = df_filtered.copy()

        df_genres['listed_in'] = df_genres['listed_in'].dropna().str.split(',')
        df_genres = df_genres.explode('listed_in')
        df_genres['listed_in'] = df_genres['listed_in'].str.strip()  # remove leading/trailing spaces

        top_genres = df_genres['listed_in'].value_counts().head(10)

        plt.figure(figsize=(10, 5))
        sns.barplot(x=top_genres.index, y=top_genres.values, palette='coolwarm')
        plt.title('Top 10 Genres by Number of Titles')
        plt.xlabel('Genre')
        plt.ylabel('Number of Titles')
        plt.xticks(rotation=45)
        plt.grid(True)

        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("---")

with tab3:
    st.subheader("Duration Analysis")
    st.caption("Analyze how movie durations and TV show seasons vary across the Netflix catalog.")

    with st.expander("See Distribution of Movie Durations"):
        st.subheader("Distribution of Movie Durations")
   
        movies = df_filtered[df_filtered['type'] == 'Movie'].copy()
        movies['duration_int'] = pd.to_numeric(movies['duration'].str.extract(r'(\d+)')[0], errors='coerce')
        movies_clean = movies.dropna(subset=['duration_int'])

        plt.figure(figsize=(10,6))
        sns.histplot(movies_clean['duration_int'], bins=30, kde=True, color='skyblue')
        plt.xlabel('Duration (minutes)')
        plt.ylabel('Number of Movies')
        plt.title('Movie Duration Distribution')
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("---")

    
    with st.expander("See TV Shows by Number of Seasons"):
        st.subheader("TV Shows by Number of Seasons")

        tv_shows = df_filtered[df_filtered['type'] == 'TV Show'].copy()
        tv_shows['duration_int'] = pd.to_numeric(tv_shows['duration'].str.extract(r'(\d+)')[0], errors='coerce')
        tv_shows_clean = tv_shows.dropna(subset=['duration_int'])

        plt.figure(figsize=(8,6))
        sns.countplot(x='duration_int', data=tv_shows_clean, color='orange')
        plt.xlabel('Number of Seasons')
        plt.ylabel('Count')
        plt.title('TV Shows Duration Distribution')
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("---")

with tab4:
    st.subheader("Text Analysis")
    st.caption("Dive into the language of Netflix titles and their release patterns with visual tools like word clouds and scatter plots.")

    with st.expander("See Word Cloud of Netflix Titles"):
        st.subheader("Word Cloud of Netflix Titles")
        if not df_filtered.empty:
             # Combine all titles into a single string
             text = " ".join(title for title in df_filtered['title'].dropna())
             # Create and generate a word cloud image
             wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

             # Display the generated image
             plt.figure(figsize=(15, 7))
             plt.imshow(wordcloud, interpolation='bilinear')
             plt.axis('off')
             plt.title('Word Cloud of Netflix Titles')
             st.pyplot(plt.gcf())
             plt.clf()

             st.markdown("---")
        else:
           st.info("No data selected. Please choose at least one filter option to generate the Word Cloud.")

    with st.expander("See Movie Duration vs Release Year (Interactive Scatter Plot)"):
        st.subheader("Movie Duration vs Release Year (Interactive Scatter Plot)")
        st.markdown("**Note:** The scatter plot below shows overall movie duration trends and is not affected by country or other filters.")
    
        movies = df[df['type'] == 'Movie'].copy()
        movies_clean = df[df['type'] == 'Movie'].copy()
        movies_clean['duration_int'] = pd.to_numeric(movies_clean['duration'].str.extract(r'(\d+)')[0], errors='coerce')
    

        #Ensure required columns are numeric and drop missing
        scatter_df = movies_clean[['release_year', 'duration_int']].dropna()

        import plotly.express as px

        fig = px.scatter(
            scatter_df,
            x='release_year',
            y='duration_int',
            title='Movies Duration vs Release Year',
            labels={'release_year': 'Release Year', 'duration_int': 'Duration (Minutes)'},
            opacity=0.6,
            color_discrete_sequence=['indianred']
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)



        st.markdown("---")

with tab5:
    st.subheader("Trends")
    st.caption("Discover trends in Netflix's content additions and popularity over time.")

    with st.expander("See Trend of Netflix Content Released Over the Years"):
         st.subheader("Trend of Netflix Content Released Over the Years")
    
         # Group by release year and count
         yearly_counts = df['release_year'].value_counts().reset_index()
         yearly_counts.columns = ['release_year', 'count']
         yearly_counts = yearly_counts.sort_values('release_year')

         fig = px.line(
             yearly_counts, 
             x='release_year', 
             y='count', 
             title='Number of Netflix Titles Released Each Year',
             labels={'release_year': 'Release Year', 'count': 'Number of Titles'}
             )

         st.plotly_chart(fig, use_container_width=True)

         st.markdown("---")

with tab6:
    st.subheader("Sentiment")
    st.caption("Analyze the average sentiment of Netflix title descriptions over the years.")

    with st.expander("See Average Sentiment of Netflix Titles Over the Years"):
        st.subheader("Average Sentiment of Netflix Titles Over the Years")
        # Initialize the sentiment analyzer
        sia = SentimentIntensityAnalyzer()

        # Apply sentiment analysis on the 'title' column, create a new 'sentiment' column with compound score
        df['sentiment'] = df['title'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
        #Visualize Sentiment Scores over Year
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')

        sentiment_by_year = df.groupby('release_year')['sentiment'].mean().reset_index()

        plt.figure(figsize=(12,6))
        sns.lineplot(data=sentiment_by_year, x='release_year', y='sentiment')
        plt.xlabel('Release Year')
        plt.ylabel('Average Sentiment (Compound Score)')
        plt.grid(True)
        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("---")

with tab7:
    st.subheader("Netflix Title Recommender")
    st.caption("Select a Netflix title to get smart recommendations for similar movies or shows.")

    with st.expander("See Netflix Title Recommender"):
        st.subheader("Netflix Title Recommender")

        # Fill missing genres with empty string
        df['listed_in'] = df['listed_in'].fillna('')

        # Combine genres into a single string (removes commas)
        df['genres_str'] = df['listed_in'].str.replace(', ', ' ')

        # Prepare TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['genres_str'])

        # Compute cosine similarity matrix
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        # Create a Series to map movie titles to DataFrame index
        indices = pd.Series(df.index, index=df['title']).drop_duplicates()

        # User selects a title from dropdown
        movie_list = df['title'].dropna().unique()
        selected_movie = st.selectbox("Choose a movie or show to get recommendations", sorted(movie_list))

        # Define recommendation function
        def get_recommendations(title, cosine_sim=cosine_sim):
            idx = indices.get(title)
            if idx is None:
                return []

            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]  # Top 10 recommendations
            movie_indices = [i[0] for i in sim_scores]

            return df['title'].iloc[movie_indices]

        # Button to get recommendations
        if st.button("Get Recommendations"):
            recommendations = get_recommendations(selected_movie)
            if not recommendations.empty:
                st.subheader("You might also like:")
                for title in recommendations:
                    st.markdown(f"- {title}")
            else:
                st.warning("No recommendations found. Try a different title.")



# In[ ]:




