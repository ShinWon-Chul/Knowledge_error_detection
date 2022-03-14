def count_element(centroids, clusters):
    count = {}
    for i in range(len(centroids)):
        count[i] = list(clusters.values()).count(i)
    count['x'] = list(clusters.values()).count('x')
    return count

def count_TPFP(centroids, true_clusters=None, false_clusters=None):
    """
    -function
    param:
    true_cluster(dictionary) --  
        - key(string) : pos entity, - value(int) : cluster label
    false_cluster(dictionary) --  
        - key(string) : neg entity, - value(int) : cluster label
    
    return: 
    true_count(dictionary) -- 
        -key : cluster number, -value : num_pos_entity
    false_count(dictionary) -- 
        -key : cluster number, -value : num_neg_entity
    """
    true_count = count_element(centroids, true_clusters)
    
    if false_clusters:
        false_count = count_element(centroids, false_clusters)
    else:
        false_count = None

    return true_count, false_count