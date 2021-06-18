import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

from functions import screen_data, scater_plot, elbow_method, cluster_data, scale_cluster_data

raw_data = pd.read_csv("iris_with_species.csv")

st.set_page_config(page_title="Species Segmentation with Cluster Analysis",page_icon="🌼", layout="wide", initial_sidebar_state="auto")

col_a, col_b = st.beta_columns([4,1])
col_a.title("Iris flower data set")
col_a.header("Species Segmentation with Cluster Analysis (KMeans)")
col_b.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Scikit_learn_logo_small.svg/1200px-Scikit_learn_logo_small.svg.png")

show_real_data = st.sidebar.checkbox("Show real data", help="Test some clusters yourself first")

width = st.sidebar.slider("Plot width", min_value=1, max_value=25, step=1, value=12, format="%i")
height = st.sidebar.slider("Plot height", min_value=1, max_value=25, step=1, value=5, format="%i")
plot_size = (width, height)

petals_or_sepals = st.sidebar.radio("Select data for analysis", ["sepal", "petal"])

scale_data_box = st.sidebar.checkbox("Scale Data", help="Standardize features by removing the mean and scaling to unit variance. For more info read here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#")

test_clusters = st.sidebar.number_input("Test clusters", value=1, min_value=1, step=1, format="%i")


data_screen = screen_data(data=raw_data, petal_sepal=petals_or_sepals)

if scale_data_box:
    data = scale_cluster_data(data=data_screen, number_of_clusters=test_clusters)
else:
    data = cluster_data(data=data_screen, number_of_clusters=test_clusters)

real_results = raw_data.copy()
real_results["clusters"] = real_results["species"].map({"setosa":0, "versicolor":1 , "virginica":2})

if show_real_data:
    col1, col2 = st.beta_columns(2)
    col1.subheader("Your analysis")
    col1.pyplot(scater_plot(data=data, value=petals_or_sepals, size=plot_size))
    
    col2.subheader("Real data")
    col2.pyplot(scater_plot(data=real_results, value=petals_or_sepals, size=plot_size))
else:
    st.pyplot(scater_plot(data=data, value=petals_or_sepals, size=plot_size))

if st.sidebar.checkbox("Get a hint with Elbow Method"):
    number_of_cluster_elbow = st.sidebar.number_input("Select number of cluster", value=5, min_value=1, step=1, format="%i", help="""Cluster number keeps track the highest number of clusters we want to use the WCSS method for.
    More info https://en.wikipedia.org/wiki/Elbow_method_(clustering)""")

    st.pyplot(elbow_method(data=data, clusters=number_of_cluster_elbow, size=plot_size))

conclusion = """
    The original dataset has 3 sub-species of the Iris flower. Therefore, the number of clusters is 3.
    Read more here: https://en.wikipedia.org/wiki/Iris_flower_data_set
    
    This shows us that:

    * the Eblow method is imperfect (we might have opted for 2 or even 4 clusters)
    * k-means is very useful in moments where we already know the number of clusters - in this case: 3
    * biology cannot be always quantified   
    """

iris = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/1280px-Iris_versicolor_3.jpg"

if show_real_data:
    with st.beta_expander("Iris flower data set"):
        col_1, col_2 = st.beta_columns([2,1])
        col_1.write(conclusion)
        col_2.image(iris)