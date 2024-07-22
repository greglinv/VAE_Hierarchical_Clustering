def calculate_jaccard_similarity(minhashes):
    num_files = len(minhashes)
    similarity_matrix = [[0] * num_files for _ in range(num_files)]
    for i in range(num_files):
        for j in range(i + 1, num_files):
            similarity = minhashes[i].jaccard(minhashes[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity
    return similarity_matrix
