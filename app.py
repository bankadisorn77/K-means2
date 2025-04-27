# import streamlit as st
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs

# with open('kmeans_model.pkl', 'rb') as f:
#     model = pickle.load(f)
# # Set page config first
# st.set_page_config(page_title='k-Means Clustering', layout='wide')

# # --- Sidebar for Configuration ---
# with st.sidebar:
#     st.header("Configure Clustering")
#     k = st.slider('Select number of Clustering (k)', 2, 10, 3)

# # --- Main Area ---
# st.title("k-Means Clustering Visualizer by Adisorn Saard")

# st.subheader('Example Data for Visualization')
# st.markdown('This demo uses example 2D data to illustrate clustering results.')

# # Generate example data
# X, _ = make_blobs(n_samples=300, centers=k, cluster_std=0.60, random_state=0)

# y_kmeans = model.fit_predict(X)

# # Plot
# fig, ax = plt.subplots()
# scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
# ax.set_title(f'k-Means Clustering (k={k})')
# ax.legend()
# st.pyplot(fig)
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns

# Set page config first
st.set_page_config(page_title='Iris k-Means Clustering', layout='wide')

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configure Clustering")
    k = st.slider('Select number of Clusters (k)', 2, 10, 3)
    random_state = st.sidebar.number_input("Random State", value=42, step=1)

# --- Main Area ---
st.title("Iris Dataset k-Means Clustering")
st.subheader('Clustering of the Iris Dataset')
st.markdown('This app applies k-Means clustering to the Iris dataset and visualizes the results using PCA for dimensionality reduction.')

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
X = iris.data
target = iris.target_names[iris.target]
df['target'] = target

st.subheader("Iris Dataset Overview")
st.dataframe(df.head())

# Train the k-Means model
kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
df['cluster'] = kmeans.fit_predict(X)

# Perform PCA for visualization (reduce to 2 dimensions)
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = df['cluster']
pca_df['target'] = df['target']

st.subheader(f'k-Means Clustering Results (k={k})')

# Create the scatter plot using Seaborn for better aesthetics
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=pca_df, palette='viridis')
sns.scatterplot(x=kmeans.cluster_centers_[:, 0], y=kmeans.cluster_centers_[:, 1],
                s=300, color='red', marker='*', label='Centroids')
ax.set_title(f'k-Means Clustering of Iris Data (2D PCA Projection)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.legend()
st.pyplot(fig)

st.subheader("Cluster Distribution")
cluster_counts = df['cluster'].value_counts().sort_index()
st.bar_chart(cluster_counts)

st.subheader("Original Target vs. Cluster Assignment")
comparison_df = pd.DataFrame({'Original Target': df['target'], 'Cluster': df['cluster']})
st.dataframe(comparison_df)
