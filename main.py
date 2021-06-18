import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

from functions import scale_data, screen_data, scater_plot, elbow_method, cluster_data

raw_data = pd.read_csv("iris_with_species.csv")

st.set_page_config(page_title="Species Segmentation with Cluster Analysis",page_icon="🌼", layout="wide", initial_sidebar_state="auto")

petal_or_sepals = st.sidebar.radio("Select data for analysis", ["Sepal", "Petal"])
scale_data_box = st.sidebar.checkbox("Scale Data", help="Standardize features by removing the mean and scaling to unit variance. For more info read here: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#")


if scale_data_box:
    data = screen_data(data=raw_data, petal_sepal=petal_or_sepals)
    data = scale_data(data=data)
else:
    data = screen_data(data=raw_data, petal_sepal=petal_or_sepals)

st.table(data)

#inputs from streamlit app

# BUFFER_SIZE = st.sidebar.number_input("Buffer Size", value=100, min_value=1, step=100, format="%i", help="")
# BATCH_SIZE = st.sidebar.number_input("Batch Size", value=100, min_value=1, step=100, format="%i", help="")
# hidden_layers_number = st.sidebar.number_input("Number of Hidden Layers", value=2, min_value=1, step=1, format="%i", help="")
# hidden_layer_size = st.sidebar.number_input("Hidden Layer Unit Size", value=50, min_value=1, step=10, format="%i", help="")
# activation_function = st.sidebar.selectbox("Select Activation Function", activation_functions_list, index=7, help="")
# model_optimizer = st.sidebar.selectbox("Select Model Optimiser Class", model_optimizer_list, index=2, help="")
# model_loss = st.sidebar.selectbox("Select Model Loss", model_loss_list, index=28, help="")
# NUM_EPOCHS = st.sidebar.number_input("Number of epochs", value=5, min_value=1, step=1, format="%i", help="")

test_clusters = st.sidebar.number_input("Test clusters", value=1, min_value=1, step=1, format="%i")


if st.sidebar.checkbox("Get help with Elbow Method"):
    number_of_cluster_elbow = st.sidebar.number_input("Select number of cluster", value=5, min_value=1, step=1, format="%i", help="""Cluster number keeps track the highest number of clusters we want to use the WCSS method for.
    More info https://en.wikipedia.org/wiki/Elbow_method_(clustering)""")

    st.pyplot(elbow_method(clusters=number_of_cluster_elbow, data=data))

