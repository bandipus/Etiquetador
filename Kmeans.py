__authors__ = ['1636012','1637892','1633445']
__group__ = 'DM.18'

import numpy as np
import utils
from sklearn.metrics import silhouette_score


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        
        X = X.astype(float)
        
        if X.ndim > 2:
            X = X.reshape(-1, 3)

        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        self.old_centroids = np.zeros((self.K, self.X.shape[1]))
        self.centroids = np.zeros((self.K, self.X.shape[1]))

        if self.options['km_init'] == 'first':
            temp_dict = {}
            for i in self.X:
                temp_dict[tuple(i)] = 1
            unique_points = np.array(list(temp_dict.keys()))
            self.centroids = unique_points[:self.K]
        elif self.options['km_init'] == 'random':
            temp_dict = {}
            for i in self.X:
                temp_dict[tuple(i)] = 1
            unique_points = np.array(list(temp_dict.keys()))
            
            np.random.shuffle(unique_points)
            self.centroids = unique_points[:self.K]
        elif self.options['km_init'] == 'kmeans++':
            centroids = [self.X[np.random.choice(self.X.shape[0])]]
            distances = np.sum((self.X - centroids[0])**2, axis=1)

            for i in range(1, self.K):
                probabilities = distances / np.sum(distances)

                new_centroid_index = np.random.choice(self.X.shape[0], p=probabilities)
                new_centroid = self.X[new_centroid_index]
                centroids.append(new_centroid)

                new_distances = np.sum((self.X - new_centroid)**2, axis=1)
                distances = np.minimum(distances, new_distances)
            
            self.centroids = np.array(centroids)
        else:
            pass

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        
        mat_distance = distance(self.X, self.centroids)
        self.labels = np.argmin(mat_distance, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        
        self.old_centroids = self.centroids

        unique_labels = np.unique(self.labels)

        self.count_dict = {label: [] for label in unique_labels}

        for i, label in enumerate(self.labels):
            point = self.X[i]
            self.count_dict[label].append(point)
        self.centroids = np.array([np.mean(self.count_dict[label], axis=0) for label in unique_labels])

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        
        if self.options["tolerance"] != "0":
            return np.allclose(self.centroids, self.old_centroids, atol=self.options["tolerance"])
        else:
            return np.allclose(self.centroids, self.old_centroids)

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        self._init_centroids()
        
        while not self.converges() and self.num_iter != self.options["max_iter"]:
            self.get_labels()
            self.get_centroids()
            self.num_iter += 1


    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        WCD = 0
        N = len(self.X)
        for label, points in self.count_dict.items():
            cx = self.centroids[label]
            diff = points - cx
            WCD += np.sum(np.linalg.norm(diff, axis=1) ** 2)
        if WCD != 0:
            WCD /= N
        return WCD
    
    def interClassDistance(self):
        """
        Calcula la distància inter-class del clustering actual
        """
        ICD = 0
        for label1 in range(self.K-1):
            for label2 in range(label1+1, self.K):
                cx1 = self.centroids[label1]
                cx2 = self.centroids[label2]
                diff = cx1 - cx2
                ICD += np.linalg.norm(diff)
                return ICD

            
    def fisherCoefficient(self):
        """
        Calcula el coeficient de Fisher del clustering actual
        """
        WCD = self.withinClassDistance()
        ICD = self.interClassDistance()
        FC = ICD / WCD
        return FC


    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        llindar = 20
        self.K = 1
        self.fit()
        lastWCD = self.withinClassDistance()
        for k in range(2, max_K+1):
            self.K = k
            self.fit()
            actualWCD = self.withinClassDistance()
            PDEC = 100 * actualWCD / lastWCD
            lastWCD = actualWCD
            if (100 - PDEC <= llindar):
                self.K = k-1
                return;
            
            
    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))
    
    
    def predict(self):
        labels = []
        for x in self.X:
            distances = [self._distance(x, centroid) for centroid in self.centroids]
            label = np.argmin(distances)
            labels.append(label)
        return labels


    def find_bestK_Improved(self, max_K):

        best_score = -1
        self.K = 1
        self.fit()
        best_K = 1
        for k in range(2, max_K + 1):
            self.K = k
            self.fit()
            labels = self.predict()
            score = silhouette_score(self.X, labels)
            if score > best_score:
                best_score = score
                best_K = k
                self.K = best_K
                
    def find_bestK_differents_heuristics(self, max_K):
        """
        Estableix la millor K analitzant els resultats fins a 'max_K' clusters
        """
        threshold = 20
        self.K = 1
        self.fit()
        lastWCD = self.withinClassDistance()
        lastICD = self.interClassDistance()
        lastFC = self.fisherCoefficient()
        
        for k in range(2, max_K+1):
            self.K = k
            self.fit()
            
            actualWCD = self.withinClassDistance()
            actualICD = self.interClassDistance()
            actualFC = self.fisherCoefficient()
            
            PDEC_WCD = 100 * actualWCD / lastWCD
            PDEC_ICD = 100 * actualICD / lastICD
            PDEC_FC = 100 * actualFC / lastFC
        
            lastWCD = actualWCD
            lastICD = actualICD
            lastFC = actualFC
            
            if (100 - PDEC_WCD <= threshold) or (100 - PDEC_ICD <= threshold) or (100 - PDEC_FC <= threshold):
                self.K = k-1
                return

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    
    distances = np.sqrt(np.sum((X[:, np.newaxis, :] - C) ** 2, axis=2))

    return distances

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    color_prob = utils.get_color_prob(centroids)

    max_index = np.argmax(color_prob, axis=1)

    color_labels = [utils.colors[i] for i in max_index]

    return color_labels
