import streamlit as st
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Set page config
st.set_page_config(page_title='k-Means Clustering', layout='centered')

# Load model
with open('kmeans_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create two columns
col1, col2 = st.columns([1, 3])

with col1:
    st.title("Configure Clustering")
    k = st.slider('Select number of Clustering (k)', 2, 10, 3)  # default should be inside the range (2-10)

with col2:
    #  Set title
    st.title("k-Means Clustering Visualizer by Adisorn Saard")

    st.subheader('Example Data for Visualization')
    # st.markdown('This demo uses example 2D data to illustrate clustering results.')
  
    # Load sample data
    X, _ = make_blobs(n_samples=300, centers=k, cluster_std=0.60, random_state=0)

    # Predict using the model
    try:
        y_kmeans = model.predict(X)

        # Plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
        ax.set_title('k-Means Clustering')
        ax.legend()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Prediction Error: {e}")
