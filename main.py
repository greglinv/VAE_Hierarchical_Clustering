from data_loader import load_fileset
from minhash import construct_minhash_representations
from similarity import calculate_jaccard_similarity
from clustering import hierarchical_clustering, vae_clustering
from error_analysis import error_analysis
import numpy as np
import time


def main():
    # Step 1: Input Fingerprints of Filesets
    path = r'D:\FSl Data\fslhomes-user000-2015-04-10'
    fingerprints = load_fileset(path)

    # Debugging statement to check fingerprints
    print("Fingerprints loaded:", fingerprints[:5])  # Print first 5 fingerprints for verification

    # Convert fingerprints to binary
    fingerprints_binary = np.array([list(bin(int(fp, 16))[2:].zfill(256)) for fp in fingerprints]).astype(int)

    # Step 2: Prephase - Construct Compact Representations using MinHash
    minhashes = construct_minhash_representations(fingerprints)

    # Step 3: Calculate Jaccard Similarity Coefficient
    similarity_matrix = calculate_jaccard_similarity(minhashes)

    # Step 4: Clustering
    start_time_hierarchical = time.time()
    clusters_hierarchical = hierarchical_clustering(similarity_matrix)
    end_time_hierarchical = time.time()

    start_time_vae = time.time()
    clusters_vae = vae_clustering(fingerprints_binary)
    end_time_vae = time.time()

    # Ground truth (example, replace with actual ground truth if available)
    ground_truth = []  # Replace with actual ground truth data

    # Step 5: Error Analysis
    results = error_analysis(fingerprints_binary, clusters_hierarchical, clusters_vae, ground_truth,
                             start_time_hierarchical, end_time_hierarchical, start_time_vae, end_time_vae)

    print("\nClustering Results:")
    print(f"Hierarchical Clustering: {clusters_hierarchical}")
    print(f"VAE-based Clustering: {clusters_vae}")
    print("\nPerformance Metrics:")
    print(f"Accuracy (Hierarchical): {results['accuracy_hierarchical']}")
    print(f"Accuracy (VAE): {results['accuracy_vae']}")
    print(f"Silhouette Score (Hierarchical): {results['silhouette_hierarchical']}")
    print(f"Silhouette Score (VAE): {results['silhouette_vae']}")
    print(f"Efficiency (Hierarchical): {results['efficiency_hierarchical']} seconds")
    print(f"Efficiency (VAE): {results['efficiency_vae']} seconds")


if __name__ == "__main__":
    main()
