import pickle
import unittest

import KNN as k
from KNN import *
from utils import *


class TestCases(unittest.TestCase):

    def setUp(self):
        np.random.seed(123)
        with open('./test/test_cases_knn.pkl', 'rb') as f:
            self.test_cases = pickle.load(f)

    def test_NIU(self):
        # DON'T FORGET TO WRITE YOUR NIU AND GROUPS
        self.assertNotEqual(k.__authors__, "TO_BE_FILLED", msg="CHANGE IT TO YOUR NIU!")
        self.assertNotEqual(k.__group__, "TO_BE_FILLED", msg="CHANGE YOUR GROUP NAME!")

    def test_init_train(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            np.testing.assert_array_equal(knn.train_data, self.test_cases['init_train'][ix])

    def test_get_k_neighbours(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            knn.get_k_neighbours(self.test_cases['test_input'][ix][0], self.test_cases['rnd_K'][ix])
            np.testing.assert_array_equal(knn.neighbors, self.test_cases['get_k_neig'][ix])

    def test_get_class(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            knn.get_k_neighbours(self.test_cases['test_input'][ix][0], self.test_cases['rnd_K'][ix])
            preds = knn.get_class()
            np.testing.assert_array_equal(preds, self.test_cases['get_class'][ix])

    def test_fit(self):
        for ix, (train_imgs, train_labels) in enumerate(self.test_cases['input']):
            knn = KNN(train_imgs, train_labels)
            preds = knn.predict(self.test_cases['test_input'][ix][0], self.test_cases['rnd_K'][ix])
            np.testing.assert_array_equal(preds, self.test_cases['get_class'][ix])


if __name__ == "__main__":
    unittest.main()
