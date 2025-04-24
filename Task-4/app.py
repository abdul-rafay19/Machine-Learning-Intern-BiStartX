import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Streamlit UI
st.title("Mall Customer Segmentation")
st.subheader("Using K-Means Clustering")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load data
if uploaded_file:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("Mall_Customers.csv")  # Make sure this file exists in the same directory

# Preprocessing
try:
    data["Gender"] = data["Gender"].map({"Male": 0, "Female": 1})
    features = data.drop(columns=["CustomerID"])
    
    scaler = StandardScaler()  # <- This was missing
    processed = scaler.fit_transform(features)

    # K-Means
    kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
    clusters = kmeans.fit_predict(processed)
    data["Cluster"] = clusters

    # Visualization
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x=data["Annual Income (k$)"], 
        y=data["Spending Score (1-100)"], 
        hue=data["Cluster"], 
        palette="Set1", 
        ax=ax
    )
    st.pyplot(fig)

    # Show clustered data
    st.subheader("Segmented Data")
    st.dataframe(data)

except Exception as e:
    st.error(f"An error occurred: {e}")
