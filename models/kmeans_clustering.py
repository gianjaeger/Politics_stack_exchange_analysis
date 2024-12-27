# models/kmeans_clustering.py

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def cluster_posts_with_pca(all_posts, tracked_users, n_clusters=4, max_features=100, random_state=42):

    # Step 1: Subset relevant columns and drop NaN
    all_posts = all_posts[['Cleaned_Body', 'OwnerUserId']].dropna()

    # Step 2: Identify posts from tracked users
    all_posts['IsTrackedUser'] = all_posts['OwnerUserId'].isin(tracked_users)

    # Step 3: TF-IDF Vectorization of 'Cleaned_Body'
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(all_posts['Cleaned_Body'])

    # Step 4: Perform K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    all_posts['Cluster'] = kmeans.fit_predict(tfidf_matrix)

    # Step 5: Dimensionality Reduction with PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(tfidf_matrix.toarray())
    all_posts['PCA1'] = reduced_features[:, 0]
    all_posts['PCA2'] = reduced_features[:, 1]

    # Step 6: Sample non-tracked data and separate tracked data
    non_tracked_data = all_posts[~all_posts['IsTrackedUser']].sample(
        n=min(10000, len(all_posts[~all_posts['IsTrackedUser']])),
        random_state=random_state
    )
    tracked_data = all_posts[all_posts['IsTrackedUser']]

    # Step 7: Transform cluster centers to PCA space
    cluster_centers = kmeans.cluster_centers_
    reduced_centers = pca.transform(cluster_centers)

    # Return results
    return {
        "all_posts": all_posts,
        "reduced_centers": reduced_centers,
        "non_tracked_data": non_tracked_data,
        "tracked_data": tracked_data,
    }