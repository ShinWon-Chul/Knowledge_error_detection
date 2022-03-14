from AdaClustering.ClusteringAlgorithm import grant_to_cluster
from util.counter import count_element, count_TPFP
import numpy as np

def get_initial_TP(pos_vectors, max_distance_each_cluster, centroids):
    
    true_clusters = grant_to_cluster(pos_vectors, 1, max_distance_each_cluster, centroids)
    true_count, _ = count_TPFP(centroids, true_clusters, None)

    initial_TP = [v for k, v in true_count.items()]
    return initial_TP

def get_precision_recall_each_cluster(true_count, false_count, cluster_labels, initial_TP):
    """
    -function
    
    param:
    true_count(dictionary) -- 
        -key : cluster number, -value : num_pos_entity
    false_count(dictionary) --
        -key : cluster number, -value : num_neg_entity
    cluster_labels(int) -- num clusters
    initial_TP(list) -- 각 cluster에 속한 초기 positive entity수
    
    return:
    """
    precision_list = []
    recall_list = []
    f1_list = []
    tp_list = []
    fn_list = []
    fp_list = []
    tn_list = []
    
    # 각 cluster를 순회하며 precision, recall, f1 score TP, FN, FP, TN을 list에 저장
    for cluster in range(cluster_labels):
        TP = true_count[cluster]
        FN = initial_TP[cluster] - TP
        FP = false_count[cluster]
        TN = false_count['x'] 

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
        tn_list.append(TN)

    return precision_list, recall_list, f1_list,\
                tp_list, fn_list, fp_list, tn_list


def get_ada_matrics(pos_vector, neg_vector, max_distance_each_cluster, centroids):
    """
    - function
    
    parameters:
    pos_vector(train_pos_vector) -- [(word, vector), ... (word, vector)]
    neg_vector(train_neg_vector) -- [(word, vector), ... (word, vector)]
    max_distance_each_cluster(dictionary) -- 
        key - cluster lable, value - max_distance
    centroids -- 클러스터의 중심점 벡터 (k, embedding size) (27,100)
    
    return:
    precision_matrix(2D numpy array) -- precision of each cluster and delta / shape(40, number of cluster)
    recall_matrix(2D numpy array) -- recall of each cluster and delta / shape(40, number of cluster)
    f1_matrix(2D numpy array) -- f1-score of each cluster and delta / shape(40, number of cluster)
    """
    
    #0.6에서 1.0까지의 0.01간격으로 변하는 모든 delta값에서 각 cluster의 matric값을 저장할 2차원 배열 
    # 행 delta값, 열 cluster number
    precision_matrix = []
    recall_matrix = []
    f1_matrix = []
    tp_matrix = []
    fn_matrix = []
    fp_matrix = []
    tn_matrix = []
    
    # true_count(dictionary) : -key : cluster number, -value : num_pos_entity
    # false_count(dictionary) : -key : cluster number, -value : num_neg_entity
    true_cluster = grant_to_cluster(pos_vector, 1, max_distance_each_cluster, centroids)
    false_cluster = grant_to_cluster(neg_vector, 1, max_distance_each_cluster, centroids)


    true_count, false_count = count_TPFP(centroids, true_cluster, false_cluster)
    initial_TP = [v for _, v in true_count.items()]

    deltas = np.arange(0.6,1.01,0.01)
    
    #델타를 조정하며 지표 계산
    for delta in deltas:
        true_cluster = grant_to_cluster(pos_vector, delta, max_distance_each_cluster, centroids)
        false_cluster = grant_to_cluster(neg_vector, delta, max_distance_each_cluster, centroids)
        true_count, false_count = count_TPFP(centroids, true_cluster, false_cluster)
        
        # 현제 delta 값에서 모든 클러스터의 precision, recall, f1 score, TP, FP, FN, TN을 구한 리스트 반환
        p, r, f1, tp, fn, fp, tn= get_precision_recall_each_cluster(true_count, false_count, len(centroids), initial_TP)

        precision_matrix.append(p) # precision of each cluster and delta / shape(40, len(centroids)) 
        recall_matrix.append(r) # recall of each cluster and delta / shape(40, len(centroids))
        f1_matrix.append(f1)
        tp_matrix.append(tp)
        fn_matrix.append(fn)
        fp_matrix.append(fp)
        tn_matrix.append(tn)

    return np.array(precision_matrix), np.array(recall_matrix), np.array(f1_matrix),\
            np.array(tp_matrix), np.array(fn_matrix),\
            np.array(fp_matrix), np.array(tn_matrix), initial_TP

def get_optimal_matrics(pos_vector, neg_vector, max_distance_each_cluster, centroids, optimaldeltas, initial_TP):
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