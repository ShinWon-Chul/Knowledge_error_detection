from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances  #k-means using euclidean_distances
import numpy as np

def kmeans_alg(entity_list, k, vector_dict):
    '''
    - function run scikit learn K-means algorithm for positive entities

    parameters:
        entity_list(list) -- list of entities (e.g. ['entity_1', 'entity_2', ..., 'entity_n'])
        k(int) -- number of centroids
        vector_dict(dictionary) -- a dictionary that maps entities to vector values
            - key(string) : entity / -value(numpy array) : entity vector

    return:
        vectors(numpy array) shape(num entity, embedding_size) -- glove vectors or skip-gram vectors 
        centroids(numpy array) shape(k, embedding_size) -- centroid points of k clusters
        c_label(numpy array) shape(num entity, ) -- label for the cluster to which each entity belongs to
    '''
    if type(vector_dict) == dict : 
        vectors = np.array([vector_dict[n.split()[0]] for n in entity_list])
    else:
        vectors = np.array([vector_dict.word_vec(n) for n in entity_list])

    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=500).fit(vectors)
    centroids = kmeans.cluster_centers_  # each c_count of center point 
    c_label = kmeans.labels_                    # [...] index is vector's index, value is in cluster
    
    return vectors, centroids, c_label

def grant_to_cluster(vec_tuple_list, delta, max_distance_each_cluster, centroids):
    '''
    - function assigns a given vector by checking which cluster it belongs to

    parameters:
        vec_tuple_list(list of tulples) -- an array of positive entity and its entity vector tuples
            (e.g. [(entity(string), vector(numpy array)), ... (entity(string), vector(numpy array))])
        delats(float) -- 1.0 use initial radius, can be optimized with values between 0.6 and 1.0
        max_distance_each_cluster(dictionary) -- radius of each cluster
            - key : cluster / value : cluster radius distance
        centroids(numpy array) shape(k, embedding_size) -- centroid points of k clusters

    return:
        in_cluster_dict(dictionary) -- dict of which cluster a given entity belongs to
            - key(string) : entity / - value(int) : cluster label
    '''
    in_cluster_dict = {}
    distance_matrix = euclidean_distances([i[1] for i in vec_tuple_list], centroids) # numpy(20000, number of cluster)
    
    for i in range(len(list(vec_tuple_list))):
        distance_with_centroid = distance_matrix[i] # (20000, num_cluster)
        close_cluster = np.argmin(distance_with_centroid) # 해당 entity와 가장 가까운 cluster
                
        delta = np.float32(delta)

        # 모든 거리를 소수점 아래 5번째 자리 까지 반올림하여 부동 소수점 오차 완화
        radius_delta = np.round(max_distance_each_cluster[close_cluster]*delta, 5)
        min_distance = np.round(min(distance_with_centroid), 5)
        
        # delta 값을 곱해 조정된 범위에 엔티티 벡터가 포함되는지 확인
        if  radius_delta >= min_distance:
            # 범위안에 포함된다면 해당 클러스터 number를 value로 할당
            in_cluster_dict[vec_tuple_list[i][0]] = close_cluster
        else:
            # 범위 안에 포함되지 않는다면 value에 'x' 할당
            in_cluster_dict[vec_tuple_list[i][0]] = 'x'
    
    return in_cluster_dict