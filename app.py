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
    k = st.slider('Select number of Clusters (k)', 2, 19, 3) # Adjusted max value to match image

# --- Main Area ---
st.title("🔍 K-Means Clustering App with Iris Dataset")

# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame
X = iris.data

# Perform K-Means clustering
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

# Perform PCA for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)
pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = df['Cluster']

# Create the scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis')
ax.set_title('Clusters (2D PCA Projection)')
ax.set_xlabel('PCA1')
ax.set_ylabel('PCA2')
ax.legend(title='Clusters')
st.pyplot(fig)
