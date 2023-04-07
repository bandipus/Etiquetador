__authors__ = '1636012, 1637892, 1633445'
__group__ = 'DM.18'

import numpy as np
import utils


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
        else:
            # Custom
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
        
        unique_labels = np.unique(self.labels)

        count_dict = {label: [] for label in unique_labels}
        points_dict = {label: [] for label in unique_labels}

        for i, label in enumerate(self.labels):
            point = self.X[i]
            count_dict[label].append(point)
            points_dict[label].append(point)

        new_centroids = np.array([np.mean(count_dict[label], axis=0) for label in unique_labels])
        self.old_centroids = self.centroids
        self.centroids = new_centroids
        
        self.points_in_every_centroid = points_dict


    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        
        return np.allclose(self.centroids, self.old_centroids)


    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        pass


    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
        WCD = 0
        N = len(self.X)
        for label, points in self.points_in_every_centroid.items():
            cx = self.centroids[label]
            diff = points - cx
            WCD += np.sum(np.linalg.norm(diff, axis=1) ** 2)
        if WCD != 0:
            WCD /= N
        print(WCD)
        return WCD


    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        for k in range(2, max_K+1):
            self.K = k
        """
        self.K =
        """


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

    distances = np.zeros((X.shape[0], C.shape[0]))
    
    for i in range(X.shape[0]):
        for j in range(C.shape[0]):
            distances[i, j] = np.sqrt(np.sum((X[i] - C[j]) ** 2))

    return distances

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return list(utils.colors)
