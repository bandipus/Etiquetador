__authors__ = '1636012'
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
        
        """
        if self.options['km_init'].lower() == 'first':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        """

def get_labels(self):
    """        Calculates the closest centroid of all points in X
    and assigns each point to the closest centroid
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    self.labels = np.random.randint(self.K, size=self.X.shape[0])


def get_centroids(self):
    """
    Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    pass


def converges(self):
    """
    Checks if there is a difference between current and old centroids
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    return True


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

    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    return np.random.rand()


def find_bestK(self, max_K):
    """
     sets the best k anlysing the results up to 'max_K' clusters
    """
    #######################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #######################################################
    pass


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

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return np.random.rand(X.shape[0], C.shape[0])


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
