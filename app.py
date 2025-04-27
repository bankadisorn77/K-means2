import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Set page config first
st.set_page_config(page_title='k-Means Clustering', layout='wide')

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configure Clustering")
    k = st.slider('Select number of Clustering (k)', 2, 10, 3)

# --- Main Area ---
st.title("k-Means Clustering Visualizer by Adisorn Saard")

st.subheader('Example Data for Visualization')
st.markdown('This demo uses example 2D data to illustrate clustering results.')

# Generate example data
X, _ = make_blobs(n_samples=300, centers=k, cluster_std=0.60, random_state=0)

# 🔥 Train a new KMeans model with k clusters
model = KMeans(n_clusters=k, random_state=0, n_init=10) # Added n_init for better convergence
y_kmeans = model.fit_predict(X)

# Plot
fig, ax = plt.subplots()
scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
           s=300, c='red', marker='*', label='Centroids')
ax.set_title(f'k-Means Clustering (k={k})')
ax.legend()
st.pyplot(fig)
