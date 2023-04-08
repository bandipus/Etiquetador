import pickle
import unittest

import Kmeans as km
from Kmeans import *
from utils import *


# unittest.TestLoader.sortTestMethodsUsing = None

class TestCases(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        with open('./test/test_cases_kmeans.pkl', 'rb') as f:
            self.test_cases = pickle.load(f)




    def test_09_find_bestK(self):
        for ix, input in enumerate(self.test_cases['input']):
            km = KMeans(input, self.test_cases['K'][ix])
            km.find_bestK(10)
            self.assertEqual(km.K, self.test_cases['bestK'][ix])



if __name__ == "__main__":
    unittest.main()
