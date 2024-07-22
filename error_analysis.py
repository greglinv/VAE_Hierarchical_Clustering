from sklearn.metrics import adjusted_rand_score, silhouette_score
import numpy as np

def calculate_accuracy(clusters, ground_truth):
    if not ground_truth:
        return None
    return adjusted_rand_score(ground_truth, clusters)

def calculate_silhouette_score(data, clusters):
    num_clusters = len(set(clusters))
    if num_clusters < 2 or num_clusters >= len(data):
        return None  # Cannot compute silhouette score for an invalid number of clusters
    return silhouette_score(data, clusters)

def calculate_efficiency(start_time, end_time):
    return end_time - start_time

def error_analysis(data, clusters_hierarchical, clusters_vae, ground_truth, start_time_hierarchical, end_time_hierarchical, start_time_vae, end_time_vae):
    accuracy_hierarchical = calculate_accuracy(clusters_hierarchical, ground_truth)
    accuracy_vae = calculate_accuracy(clusters_vae, ground_truth)
    silhouette_hierarchical = calculate_silhouette_score(data, clusters_hierarchical)
    silhouette_vae = calculate_silhouette_score(data, clusters_vae)
    efficiency_hierarchical = calculate_efficiency(start_time_hierarchical, end_time_hierarchical)
    efficiency_vae = calculate_efficiency(start_time_vae, end_time_vae)
    return {
        "accuracy_hierarchical": accuracy_hierarchical,
        "accuracy_vae": accuracy_vae,
        "silhouette_hierarchical": silhouette_hierarchical,
        "silhouette_vae": silhouette_vae,
        "efficiency_hierarchical": efficiency_hierarchical,
        "efficiency_vae": efficiency_vae
    }
