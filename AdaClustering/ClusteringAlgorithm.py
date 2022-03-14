from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances  #k-means using euclidean_distances
import numpy as np

def kmeans_alg(vector_list, k, glove_dict):
    '''
    function run scikit learn K-means algorithm for 20000 positive entities

    parameters:
        vector_list(list) -- list of entities to which the K-means algorithm applies
        k(int) -- number of centroids to run K-means
        glove_dict(dictionary) -- a dictionary that maps entities to vector values
            - key(string) : entity / -value(numpy array) : entity vector

    return:
        X(numpy array) shape(num entity, embedding_size) -- glove_vectors of input shape
        k_centroid_points(numpy array) shape(k, embedding_size) -- centroid points of k clusters
        k_labels(numpy array) shape(num entity, ) -- label for the cluster to which each entity belongs to
    '''
    X = np.array([glove_dict[n.split()[0]] for n in vector_list])

    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=1000).fit(X)
    k_centroid_points = kmeans.cluster_centers_  # each c_count of center point 
    k_labels = kmeans.labels_                    # [...] index is vector's index, value is in cluster
    
    return X, k_centroid_points, k_labels

def grant_to_cluster(vec_tuple_list, delta, max_distance_each_cluster, centroids):
    '''
    - function assigns a given vector by checking which cluster it belongs to

    parameters:
        vec_tuple_list(list of tulple) -- 20000 length list of tuple(entitiy, vector)
        delats(float) -- 1 use initial radius
        max_distance_each_cluster(dictionary) -- radius of each cluster dict 
            - key : cluster / value : cluster radius distance
        centroids(numpy array) shape(k, embedding_size) -- centroid points of k clusters

    return:
        in_cluster_dict(dictionary) -- dict of which cluster a given vector belongs to
            - key(string) : entity / value(int) : cluster label
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