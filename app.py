import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

st.title('Spotify Songs Data Cleaning, Exploration & Playlist Clustering')

# Use session_state to track if the workflow has started
if 'started' not in st.session_state:
    st.session_state['started'] = False

if not st.session_state['started']:
    if st.button('Start Data Cleaning and Visualization'):
        st.session_state['started'] = True
        st.rerun()
else:
    # Step 1: Load the data
    csv_path = os.path.join('data', '3_spotify_5000_songs.csv')
    st.info('Loading data from CSV file...')
    time.sleep(1)
    df = pd.read_csv(csv_path)
    st.success('Data loaded successfully!')
    time.sleep(1)

    # Step 2: Clean column names
    st.info('Cleaning column names (removing white spaces)...')
    time.sleep(1)
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    st.success('Column names cleaned!')
    time.sleep(1)

    # Step 3: Remove non-numeric columns and drop "Unnamed: 0" if present
    st.info('Removing non-numeric columns and dropping "Unnamed: 0" if present...')
    time.sleep(1)
    if 'Unnamed:_0' in df.columns:
        df = df.drop(columns=['Unnamed:_0'])
    numeric_df = df.select_dtypes(include=[np.number])
    st.success(f'Removed non-numeric columns. Remaining columns: {list(numeric_df.columns)}')
    time.sleep(1)

    # Step 4: Plot histograms for each numeric column in a 3-column grid
    st.info('Plotting histograms for each numeric column in a 3-column grid...')
    time.sleep(1)
    num_cols = len(numeric_df.columns)
    cols_per_row = 3
    rows = (num_cols + cols_per_row - 1) // cols_per_row

    for row in range(rows):
        cols = st.columns(cols_per_row)
        for i in range(cols_per_row):
            col_idx = row * cols_per_row + i
            if col_idx < num_cols:
                col_name = numeric_df.columns[col_idx]
                with cols[i]:
                    st.subheader(f'{col_name}')
                    fig, ax = plt.subplots()
                    ax.hist(numeric_df[col_name].dropna(), bins=30, color='skyblue', edgecolor='black')
                    ax.set_xlabel(col_name)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
    st.success('All histograms plotted!')

    st.header('Step 1: Feature Selection')
    selected_features = st.multiselect('Select features to use for clustering:', options=list(numeric_df.columns), default=list(numeric_df.columns))
    filtered_df = numeric_df[selected_features]

    st.header('Step 2: Choose a Scaler')
    scaler_option = st.radio('Select a scaler:', options=['StandardScaler', 'MinMaxScaler', 'RobustScaler'])
    if scaler_option == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_option == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
    scaled_data = scaler.fit_transform(filtered_df)
    st.success(f'Data scaled using {scaler_option}.')

    st.header('Step 3: PCA Dimensionality Reduction')
    variance = st.slider('Select the percentage of variance to preserve:', min_value=0.5, max_value=1.0, value=0.9, step=0.01)
    pca = PCA(n_components=variance)
    pca_data = pca.fit_transform(scaled_data)
    st.success(f'PCA applied. Number of components: {pca.n_components_}')
    st.write('Explained variance ratio:', np.round(pca.explained_variance_ratio_, 3))

    # Clustering model selection
    st.header('Step 4: Clustering Model Selection')
    clustering_options = ['KMeans', 'DBSCAN', 'AgglomerativeClustering']
    if HDBSCAN_AVAILABLE:
        clustering_options.append('HDBSCAN')
    model_choice = st.selectbox('Choose clustering model:', clustering_options)

    # Model-specific parameter controls
    clustering_kwargs = {}
    show_inertia = False
    if model_choice == 'KMeans':
        n_clusters = st.slider('Select number of clusters (playlists):', min_value=2, max_value=50, value=5)
        clustering_kwargs['n_clusters'] = n_clusters
        show_inertia = True
    elif model_choice == 'DBSCAN':
        eps = st.slider('DBSCAN: eps (neighborhood size)', min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        min_samples = st.slider('DBSCAN: min_samples', min_value=1, max_value=20, value=5)
        clustering_kwargs['eps'] = eps
        clustering_kwargs['min_samples'] = min_samples
    elif model_choice == 'AgglomerativeClustering':
        n_clusters = st.slider('Agglomerative: number of clusters', min_value=2, max_value=50, value=5)
        linkage = st.selectbox('Agglomerative: linkage', ['ward', 'complete', 'average', 'single'])
        clustering_kwargs['n_clusters'] = n_clusters
        clustering_kwargs['linkage'] = linkage
    elif model_choice == 'HDBSCAN':
        min_cluster_size = st.slider('HDBSCAN: min_cluster_size', min_value=2, max_value=50, value=5)
        clustering_kwargs['min_cluster_size'] = min_cluster_size

    # Inertia (elbow) plot for KMeans only
    if show_inertia:
        st.header('KMeans Inertia Plot (Elbow Method)')
        inertia_range = st.slider('Select range for number of clusters (elbow plot):', min_value=2, max_value=50, value=(2, 10))
        inertias = []
        cluster_range = range(inertia_range[0], inertia_range[1] + 1)
        for k in cluster_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pca_data)
            inertias.append(kmeans.inertia_)
        fig, ax = plt.subplots()
        ax.plot(list(cluster_range), inertias, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Inertia')
        ax.set_title('KMeans Inertia (Elbow Method)')
        st.pyplot(fig)

    # Fit the selected clustering model
    st.header('Step 5: Fit Clustering Model')
    clusters = None
    model = None
    silhouette = None
    if model_choice == 'KMeans':
        model = KMeans(n_clusters=clustering_kwargs['n_clusters'], random_state=42)
        clusters = model.fit_predict(pca_data)
        # Silhouette score for KMeans
        n_clusters_found = len(set(clusters))
        if n_clusters_found > 1:
            silhouette = silhouette_score(pca_data, clusters)
            st.info(f'Silhouette score (KMeans): {silhouette:.3f}')
        else:
            st.info('Silhouette score not available: KMeans found less than 2 clusters.')
    elif model_choice == 'DBSCAN':
        model = DBSCAN(eps=clustering_kwargs['eps'], min_samples=clustering_kwargs['min_samples'])
        clusters = model.fit_predict(pca_data)
    elif model_choice == 'AgglomerativeClustering':
        model = AgglomerativeClustering(n_clusters=clustering_kwargs['n_clusters'], linkage=clustering_kwargs['linkage'])
        clusters = model.fit_predict(pca_data)
        # Silhouette score for AgglomerativeClustering
        n_clusters_found = len(set(clusters))
        if n_clusters_found > 1:
            silhouette = silhouette_score(pca_data, clusters)
            st.info(f'Silhouette score (AgglomerativeClustering): {silhouette:.3f}')
        else:
            st.info('Silhouette score not available: AgglomerativeClustering found less than 2 clusters.')
    elif model_choice == 'HDBSCAN' and HDBSCAN_AVAILABLE:
        model = hdbscan.HDBSCAN(min_cluster_size=clustering_kwargs['min_cluster_size'])
        clusters = model.fit_predict(pca_data)
    else:
        clusters = np.zeros(pca_data.shape[0], dtype=int)

    st.success(f'{model_choice} clustering applied.')

    # Show cluster assignment counts
    st.write('Cluster assignment counts:')
    st.write(pd.Series(clusters).value_counts().sort_index())

    # Optionally, show a 2D or 3D scatter plot if possible
    n_pca = pca_data.shape[1]
    if n_pca >= 3:
        st.subheader('3D PCA Scatter Plot (Select Components)')
        pca_indices = list(range(n_pca))
        comp1 = st.selectbox('Select PCA component for X axis:', pca_indices, index=0, format_func=lambda x: f'PCA {x+1}')
        comp2 = st.selectbox('Select PCA component for Y axis:', pca_indices, index=1, format_func=lambda x: f'PCA {x+1}')
        comp3 = st.selectbox('Select PCA component for Z axis:', pca_indices, index=2, format_func=lambda x: f'PCA {x+1}')
        if len(set([comp1, comp2, comp3])) == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(pca_data[:, comp1], pca_data[:, comp2], pca_data[:, comp3], c=clusters, cmap='tab10', alpha=0.6)
            ax.set_xlabel(f'PCA {comp1+1}')
            ax.set_ylabel(f'PCA {comp2+1}')
            ax.set_zlabel(f'PCA {comp3+1}')
            legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
            ax.add_artist(legend1)
            st.pyplot(fig)
        else:
            st.warning('Please select three different PCA components for the 3D plot.')
    elif n_pca == 2:
        st.subheader('2D PCA Scatter Plot (Select Components)')
        pca_indices = [0, 1]
        comp1 = st.selectbox('Select PCA component for X axis:', pca_indices, index=0, format_func=lambda x: f'PCA {x+1}')
        comp2 = st.selectbox('Select PCA component for Y axis:', pca_indices, index=1, format_func=lambda x: f'PCA {x+1}')
        if comp1 != comp2:
            fig, ax = plt.subplots()
            scatter = ax.scatter(pca_data[:, comp1], pca_data[:, comp2], c=clusters, cmap='tab10', alpha=0.6)
            ax.set_xlabel(f'PCA {comp1+1}')
            ax.set_ylabel(f'PCA {comp2+1}')
            legend1 = ax.legend(*scatter.legend_elements(), title="Cluster")
            ax.add_artist(legend1)
            st.pyplot(fig)
        else:
            st.warning('Please select two different PCA components for the 2D plot.')

    # Show two random songs from each cluster with Spotify links
    st.header('Sample Songs from Each Cluster')
    # Add clusters to the original dataframe (use a copy to avoid SettingWithCopyWarning)
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    # Use the 'html' column for Spotify URLs
    url_col = 'html' if 'html' in df_with_clusters.columns else None
    # Try to find a column with song names
    name_col = None
    for col in df_with_clusters.columns:
        if 'name' in col.lower() or 'title' in col.lower() or 'track' in col.lower():
            name_col = col
            break
    for cluster_id in sorted(df_with_clusters['cluster'].unique()):
        st.subheader(f'Cluster {cluster_id}')
        cluster_songs = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
        sample_songs = cluster_songs.sample(n=min(5, len(cluster_songs)), random_state=42)
        for idx, row in sample_songs.iterrows():
            song_name = row[name_col] if name_col else f'Song {idx}'
            url = row[url_col] if url_col else None
            if url:
                st.markdown(f'- [{song_name}]({url})')
            else:
                st.markdown(f'- {song_name}') 