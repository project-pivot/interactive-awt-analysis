import streamlit as st
import pandas as pd
import altair as alt
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

"""
# Let's analyse your AWT data!
"""

# Sidebar for accepting input parameters
with st.sidebar:
    # Load AWT data
    st.header('Upload your data')
    st.markdown('**1. AWT data**')
    awt_uploaded_file = st.file_uploader("Upload your Tockler data here. You can export your data by going to Tockler > Search > Set a time period > Export to CSV.")

# Main section for processing AWT data
if awt_uploaded_file is not None:
    st.header('First glance')
    st.markdown('Ok, great! You have uploaded some data. In the table below, we show the most occurring window titles.') 
    
    # Explicitly set the delimiter as semicolon
    dataframe_awt = pd.read_csv(awt_uploaded_file, delimiter=';')

    with st.expander('Most occurring titles', expanded=True):
        st.markdown('Please provide a title for the project or case you were working on in the final column. If the title is too generic, it is fine to leave it empty.')

        # Create a combined column of 'App' and 'Title'
        dataframe_awt['App and title'] = dataframe_awt['App'] + " - " + dataframe_awt['Title']

        # Count the occurrences of each app title
        app_title_counts = dataframe_awt['App and title'].value_counts()

        # Add the slider at the beginning to control the number of titles shown
        top_number = st.slider("Optionally, you can change the number of titles shown here", 5, 200, 20)

        # Get the top 'top_number' app titles based on the slider value
        top_app_titles = app_title_counts.head(top_number).reset_index()
        top_app_titles.columns = ['App and title', 'Count']

        # Create a DataFrame with additional columns for interactivity
        interactive_df = top_app_titles.copy()
        interactive_df['Title of project or case'] = ''  # Add a notes column for user input

        # Display the interactive data editor
        edited_df = st.data_editor(interactive_df, use_container_width=True, hide_index=True, column_order=("Count", "App and title", "Title of project or case"))
        
        # Filter edited_df for rows where 'Title of project or case' is not empty
        non_empty_projects = edited_df[edited_df['Title of project or case'] != '']

        # Create a dictionary to map App_Title to 'Title of project or case'
        title_mapping = dict(zip(non_empty_projects['App and title'], non_empty_projects['Title of project or case']))
    
    with st.expander('Visualisation', expanded=True):
        # Ensure that 'Title of project or case' exists in dataframe_awt or initialize it
        if 'Title of project or case' not in dataframe_awt.columns:
            dataframe_awt['Title of project or case'] = ''

        # Update dataframe_awt with the new titles where applicable
        dataframe_awt['Title of project or case'] = dataframe_awt['App and title'].map(title_mapping).fillna('')
        
        # Prepare data for the scatter plot
        dataframe_awt['Date'] = pd.to_datetime(dataframe_awt['Begin']).dt.date
        dataframe_awt['Time'] = pd.to_datetime(dataframe_awt['Begin']).dt.time

        # Filter the dataframe to only include rows where 'Title of project or case' has been updated by the user
        # Ensure that 'Title of project or case' is non-empty in both the interactive and the main dataframe
        plot_df = dataframe_awt[dataframe_awt['App and title'].isin(non_empty_projects['App and title'])]

        # Main section for processing AWT data
        if not plot_df.empty:
            st.markdown('The scatterplot below will show your work on the specified projects or cases across the days in the provided dataset. It will automatically update every time you add a new project or case title.')

            # Create the scatter plot with Altair
            scatter_chart = alt.Chart(plot_df).mark_circle(size=20).encode(
                x=alt.X('Date:T', title='Date', axis=alt.Axis(format='%Y-%m-%d')),  # Format the x-axis to show only dates
                y=alt.Y('hoursminutes(Begin):T', title='Time of Day'),
                color=alt.Color('Title of project or case:N', legend=alt.Legend(title="Case")),
                tooltip=['App and title', 'Title of project or case', 'Begin']
            ).properties(
                width=800,
                height=800,
                title="Work on the projects/cases over time"
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=14
            ).configure_legend(
                titleFontSize=14,
                labelFontSize=12
            )

            # Display the scatter plot
            st.altair_chart(scatter_chart, use_container_width=True)

    with st.expander('Cluster titles'):

        titles = dataframe_awt['App and title']

        # Preprocess the text data
        # You can use more sophisticated methods, including lemmatization, removing stop words, etc.
        titles = titles.str.lower()

        # Vectorize the text data using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(titles)

        # Clustering with K-Means
        num_clusters = 5  # Choose the number of clusters
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        dataframe_awt['Cluster'] = kmeans.fit_predict(X)

        # Now, 'dataframe_awt' has a new column 'Cluster' indicating the cluster assignment
        dataframe_awt