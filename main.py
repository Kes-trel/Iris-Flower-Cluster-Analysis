import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing

raw_data = pd.read_csv("iris_with_species.csv")

st.set_page_config(page_title="Species Segmentation with Cluster Analysis",page_icon="🌼", layout="wide", initial_sidebar_state="auto")



