# Moozik: Spotify Song Clustering & Playlist Explorer

A Streamlit web app for exploring, clustering, and generating playlists from a dataset of 5,000 Spotify songs. The app allows you to clean, visualize, and cluster songs using various machine learning techniques, and sample songs from each cluster to check playlist quality directly on Spotify.

## Features

- **Data Cleaning & Exploration**
  - Removes whitespace from column names
  - Drops non-numeric columns and unnecessary index columns
  - Plots histograms for all numeric features
- **Interactive ML Workflow**
  - Feature selection via checkboxes
  - Choice of scaler: StandardScaler, MinMaxScaler, RobustScaler
  - PCA for dimensionality reduction (choose variance to preserve)
  - Choice of clustering model: KMeans, DBSCAN, AgglomerativeClustering, HDBSCAN (if installed)
  - Model-specific parameter tuning (e.g., number of clusters, eps, linkage)
  - KMeans inertia (elbow) plot
  - Silhouette score for KMeans and AgglomerativeClustering
- **Visualization**
  - 2D or 3D PCA scatter plot (choose which components to plot)
- **Playlist Sampling**
  - For each cluster, displays 5 random songs with clickable links to Spotify

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   If you want to use HDBSCAN, also run:
   ```bash
   pip install hdbscan
   ```
3. **Add your data**
   - Place `3_spotify_5000_songs.csv` in the `data/` folder (or update the path in `app.py` if needed).

## Usage

Run the app locally:
```bash
streamlit run app.py
```

Follow the on-screen workflow:
1. Start data cleaning and visualization
2. Select features and scaling method
3. Reduce dimensions with PCA
4. Choose and tune a clustering model
5. Explore clusters, visualize results, and sample playlists

## Requirements
- Python 3.8+
- streamlit
- pandas
- numpy
- matplotlib
- scikit-learn
- hdbscan (optional, for HDBSCAN clustering)

## License
MIT License

---

**Enjoy exploring and generating Spotify playlists with machine learning!** 