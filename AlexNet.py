import math
import numpy as np

def mAP(q_name, sorted_pool):
    ret_classes = [sorted_pool[i][0].split("_")[0:3] for i in range(len(sorted_pool))]
    q_class = q_name.split(".")[0].split("_")[0:3]
    #print(ret_classes)
    #print(q_class)
    initlist = [int(q_class == i) for i in ret_classes]
    #print(initlist)
    den = np.sum(initlist)
    #print(den)
    if den == 0:
        return 0
    x = 0
    preclist = [0]*len(initlist)
    for idx, pts in enumerate(initlist):
        x += pts #rel(n)
        preclist[idx] = x/(idx+1) #rel(n)/k
    #print(preclist)
    num = np.dot(preclist, initlist)
    #print(num)
    #print(num/den)
    return num/den

def nDCG(relevance_list, ideal_relevance_list):
    """
    Calculates the Normalized Discounted Cumulative Gain (NDCG) of a ranked list of documents.

    Args:
        relevance_list: A list of relevance scores for the documents in the ranked list.
        ideal_relevance_list: A list of ideal relevance scores for the documents in the ranked list.

    Returns:
        The NDCG of the ranked list.
    """

    # Calculate the cumulative gain of the ranked list.
    cumulative_gain = 0.0
    for id, relevance in enumerate(relevance_list):
        numerator = 2**relevance - 1
        # add 1 because python 0-index
        denominator =  np.log2(id + 2) 
        cumulative_gain += numerator/denominator
    #print(cumulative_gain)

    # Calculate the ideal cumulative gain.
    ideal_cumulative_gain = 0.0
    for id, ideal_relevance in enumerate(ideal_relevance_list):
        numerator = 2**ideal_relevance - 1
        # add 1 because python 0-index
        denominator =  np.log2(id + 2)
        #print('1', numerator) 
        ideal_cumulative_gain += numerator/denominator
    #print(ideal_cumulative_gain)

    # Normalize the cumulative gain by the ideal cumulative gain.
    if ideal_cumulative_gain==0:
        return 0
    else:
        ndcg = cumulative_gain / ideal_cumulative_gain
        return ndcg


