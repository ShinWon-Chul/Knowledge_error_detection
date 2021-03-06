from AdaClustering.ClusteringAlgorithm import grant_to_cluster
from util.counter import count_element, count_TPFP
import numpy as np

def get_initial_TP(pos_vectors, max_distance_each_cluster, centroids):
    """
    - function to get the distribution of entities in the initial cluster with delta equal to 1.

    parameters:
        pos_vectors(list of tulples) -- an array of positive entity and its entity vector tuples
                                        (e.g. [(entity(string), vector(numpy array)), ... (entity(string), vector(numpy array))])
        max_distance_each_cluster(dictionary) -- radius of each cluster 
                                               - key : cluster / - value : cluster radius distance
        centroids(numpy array) shape(k, embedding_size) -- centroid points of k clusters

    return:
        initial_TP(list) -- An array containing the number of positive entities included in each cluster.
    """
    
    true_clusters = grant_to_cluster(pos_vectors, 1.0, max_distance_each_cluster, centroids)
    true_count, _ = count_TPFP(centroids, true_clusters, None)

    initial_TP = [v for k, v in true_count.items()]
    return initial_TP

def get_precision_recall_each_cluster(true_count, false_count, cluster_labels, initial_TP):
    """
    - function that calculates the precision, recall, and f1 score of each cluster.
    
    param:
    true_count(dictionary) -- a dictionary that stores the number of positive entity belonging to each cluster.
                            - key : cluster number / - value : num_pos_entity
    false_count(dictionary) -- a dictionary that stores the number of negitive entity belonging to each cluster.
                             - key : cluster number / - value : num_neg_entity
    cluster_labels(int) -- num clusters
    initial_TP(list) -- When delta is 1, the number of initial positive entity in each cluster
    
    return:
    precision_list(list) -- an array containing the precision values for each cluster.
    recall_list(list) -- an array containing the recall values for each cluster.
    f1_list(list) -- an array containing the f1-scores for each cluster.
    tp_list(list) -- an array containing the number of positive entity for each cluster.
    fn_list(list) -- an array containing The number of positive entities 
                     that went out of the cluster range when the delta was adjusted.
    fp_list(list) -- an array containing the number of negative entity for each cluster.
    """
    precision_list = []
    recall_list = []
    f1_list = []
    tp_list = []
    fn_list = []
    fp_list = []
    
    # ??? cluster??? ???????????? precision, recall, f1 score TP, FN, FP, TN??? list??? ??????
    for cluster in range(cluster_labels):
        TP = true_count[cluster]
        FN = initial_TP[cluster] - TP
        FP = false_count[cluster]

        if (TP+FP) == 0:
            precision = 0.0
        else:
            precision = float(TP)/(TP+FP)

        if (TP+FN) == 0:
            recall = 0.0
        else:
            recall =float(TP)/(TP+FN)

        if (precision+recall) == 0:
            f1_score = 0.0
        else:
            f1_score = float(2*precision*recall) / (precision+recall)
            
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)

        tp_list.append(TP)
        fn_list.append(FN)
        fp_list.append(FP)

    return precision_list, recall_list, f1_list,\
                tp_list, fn_list, fp_list


def get_ada_matrics(pos_vector, neg_vector, max_distance_each_cluster, centroids):
    """
    - function that adjusts the delta from 6.0 to 1.0 in 0.01 steps 
      and calculates the precision, recall, and f1 score for each cluster.
      For delta, the optimal delta that maximizes the f1-scroe of each cluster is selected among a total of 41 values.
      The optimal delta selected is multiplied by the radius distance of each cluster.
    
    parameters:
    pos_vector(list of tuples) -- an array of positive entity and its entity vector tuples
                  (e.g. [(entity(string), vector(numpy array)), ... (entity(string), vector(numpy array))])
    neg_vector(list of tuples) -- an array of negative entity and its entity vector tuples
                  (e.g. [(entity(string), vector(numpy array)), ... (entity(string), vector(numpy array))])
    max_distance_each_cluster(dictionary) -- radius of each cluster dict 
                                           - key : cluster number / - value : cluster radius distance
    centroids(numpy array) shape(k, embedding_size) -- centroid points of k clusters
    
    return:
    precision_matrix(2D numpy array) shape(41, number of cluster) -- precision of each cluster and delta
    recall_matrix(2D numpy array) shape(41, number of cluster) -- recall of each cluster and delta 
    f1_matrix(2D numpy array) shape(41, number of cluster) -- f1-score of each cluster and delta
    tp_matrix(2D numpy array) shape(41, number of cluster) -- True Positive of each cluster and delta
    fn_matrix(2D numpy array) shape(41, number of cluster) -- False Negative of each cluster and delta
    fp_matrix(2D numpy array) shape(41, number of cluster) -- False Positive of each cluster and delta
    initial_TP(list) -- When delta is 1, the number of initial positive entity in each cluster
    """
    
    #0.6?????? 1.0????????? 0.01???????????? ????????? ?????? delta????????? ??? cluster??? matric?????? ????????? 2?????? ?????? 
    # ??? delta???, ??? cluster number
    precision_matrix = []
    recall_matrix = []
    f1_matrix = []
    tp_matrix = []
    fn_matrix = []
    fp_matrix = []

    # true_count(dictionary) : -key : cluster number, -value : num_pos_entity
    # false_count(dictionary) : -key : cluster number, -value : num_neg_entity
    true_cluster = grant_to_cluster(pos_vector, 1.0, max_distance_each_cluster, centroids)
    false_cluster = grant_to_cluster(neg_vector, 1.0, max_distance_each_cluster, centroids)


    true_count, false_count = count_TPFP(centroids, true_cluster, false_cluster)
    initial_TP = [v for _, v in true_count.items()]

    deltas = np.arange(0.6,1.01,0.01)
    
    #????????? ???????????? ?????? ??????
    for delta in deltas:
        true_cluster = grant_to_cluster(pos_vector, delta, max_distance_each_cluster, centroids)
        false_cluster = grant_to_cluster(neg_vector, delta, max_distance_each_cluster, centroids)
        true_count, false_count = count_TPFP(centroids, true_cluster, false_cluster)
        
        # ?????? delta ????????? ?????? ??????????????? precision, recall, f1 score, TP, FP, FN ??? ?????? ????????? ??????
        p, r, f1, tp, fn, fp = get_precision_recall_each_cluster(true_count, false_count, len(centroids), initial_TP)

        precision_matrix.append(p) # precision of each cluster and delta / shape(40, len(centroids)) 
        recall_matrix.append(r) # recall of each cluster and delta / shape(40, len(centroids))
        f1_matrix.append(f1)
        tp_matrix.append(tp)
        fn_matrix.append(fn)
        fp_matrix.append(fp)

    return np.array(precision_matrix), np.array(recall_matrix), np.array(f1_matrix),\
            np.array(tp_matrix), np.array(fn_matrix),\
            np.array(fp_matrix), initial_TP


def get_optimal_matrics(pos_vector, neg_vector, max_distance_each_cluster, centroids, optimaldeltas, initial_TP):
    """
    - function function that calculates precision, recall, TP, and FP for each cluster 
      by applying the obtained optimal delta to the cluster
    
    parameters:
    pos_vector(list of tuples) -- an array of positive entity and its entity vector tuples
                      (e.g. [(entity(string), vector(numpy array)), ... (entity(string), vector(numpy array))])
    neg_vector(list of tuples) -- an array of negative entity and its entity vector tuples
                      (e.g. [(entity(string), vector(numpy array)), ... (entity(string), vector(numpy array))])
    max_distance_each_cluster(dictionary) -- radius of each cluster dict 
                                           - key : cluster number / value : cluster radius distance
    centroids(numpy array) shape(k, embedding_size) -- centroid points of k clusters
    optimaldeltas(numpy array) shape(k,) -- an array with an optimal delta that maximizes the f1-score of each cluster.
    initial_TP(list) -- When delta is 1, the number of initial positive entity in each cluster
    
    return:
    optimalP(list) -- array with precision values calculated when optimal delta is applied to each cluster
    optimalR(list) -- array with recall values calculated when optimal delta is applied to each cluster
    optimalTP(list) -- array with the number of positive entities included when the optimal delta is applied to each cluster
    optimalFP(list) -- array with the number of negative entities included when the optimal delta is applied to each cluster
    """
    optimalP = []
    optimalR = []
    optimalTP = []
    optimalFP = []
    
    for cluster, delta in enumerate(optimaldeltas):
        True_cluster = grant_to_cluster(pos_vector,delta,max_distance_each_cluster,centroids)
        False_cluster = grant_to_cluster(neg_vector,delta,max_distance_each_cluster,centroids)
        Truecount, Falsecount = count_TPFP(centroids, True_cluster, False_cluster)
        
        #TP
        TP = []            
        for k,v in Truecount.items():
            if type(k) != str and int(k) == cluster:
                optimalTP.append(v)
            TP.append(v)
            
        #FP
        FP = []
        for k,v in Falsecount.items():
            if type(k) != str and int(k) == cluster:
                optimalFP.append(v)        
            FP.append(v)
            
        #precision  
        precisionList = []
        for i in range(len(centroids)):
            if (TP[i]+FP[i]) == 0:
                precision = float(0.0)
            else:
                precision = TP[i] / (TP[i]+FP[i])
            precisionList.append(precision)
        
        #recall
        recallList = []
        for i in range(len(centroids)):
            FN = initial_TP[i]-TP[i]
            if (TP[i]+ FN)== 0:
                recall = float(0.0)
            else:
                recall = TP[i] / (TP[i]+FN)
            recallList.append(recall)

        optimalP.append(precisionList[cluster])
        optimalR.append(recallList[cluster])
        
    return optimalP, optimalR, optimalTP, optimalFP