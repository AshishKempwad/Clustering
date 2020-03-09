class Cluster:
    def __init__(self):
        self.k=5
        
    def cluster(self,path):
        import numpy as np
        import os
        pass_random_seed=2
        Pseudo_random_generator= np.random.RandomState(pass_random_seed)
        directories = os.listdir( path )
        text_file=[]
        text_traced_data=[]
        
        #Read all text files
        for i in directories:
            with open(path + i,'rb') as f:
                temp = f.read().decode('unicode_escape')
                temp = "".join(i for i in temp if ord(i) <128)
                interm=temp
                text_file.append(interm)
        text_file = np.array(text_file)
        
        from sklearn.metrics import pairwise_distances_argmin
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        #Text Vectorizer
        tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',max_features = 20000)
        Vectorized_data = tf_idf_vectorizor.fit_transform(text_file)
        vectorized_list= Vectorized_data.toarray()
        #Clustering algorithm
        
        i = Pseudo_random_generator.permutation(vectorized_list.shape[0])[:self.k]
        centers = vectorized_list[i]
        iterations=500
        for i in range(iterations):
            cluster_labels=pairwise_distances_argmin(vectorized_list, centers)
            centroids=np.array([vectorized_list[cluster_labels == i].mean(0) for i in range(self.k)])
        return cluster_labels
    
    
    
    
    
    
    #The above code can be run by the code given below :-
    '''
    
from q6 import Cluster as cl
cluster_algo = cl()
# You will be given path to a directory which has a list of documents. You need to return a list of cluster labels for those documents
predictions = cluster_algo.cluster('./Datasets/q6/') 
'''
