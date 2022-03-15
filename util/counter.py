def count_element(centroids, clusters):
    count = {}
    for i in range(len(centroids)):
        count[i] = list(clusters.values()).count(i)
    count['x'] = list(clusters.values()).count('x')
    return count

def count_TPFP(centroids, true_clusters=None, false_clusters=None):
    """
    - function
    param:
    centroids(numpy array) shape(k, embedding_size) -- centroid points of k clusters
    true_cluster(dictionary) -- 
        a dictionary containing information about the cluster to which a given positive entity belongs.
        - key(string) : positive entity / - value(int) : cluster label 
    false_cluster(dictionary) --
        a dictionary containing information about the cluster to which a given negative entity belongs.
        - key(string) : negative entity / - value(int) : cluster label 
    
    return: 
    true_count(dictionary) -- 
        dictionary with information about the number of positive entity belonging to each cluster
        -key : cluster number, -value : the number of positive entity
    false_count(dictionary) -- 
        dictionary with information about the number of negative entity belonging to each cluster
        -key : cluster number, -value : the number of negative entity
    """
    true_count = count_element(centroids, true_clusters)
    
    if false_clusters:
        false_count = count_element(centroids, false_clusters)
    else:
        false_count = None

    return true_count, false_count