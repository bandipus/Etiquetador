__authors__ = ['1636012','1637892','1633445']
__group__ = 'DM.18'

import Kmeans as km
import numpy as np
from Kmeans import *
import KNN as knn
from KNN import *
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
import matplotlib.pyplot as plt
import time

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    def Retrieval_by_color(imgs_list, labels_kmeans, search_string, imgs_shapes=None, imgs_colors=None):
        list = []
        infoList = []
        truefalse = []
        for i, label in enumerate(labels_kmeans):
            if any(cl.lower() == search_string.lower() for cl in label):
                list.append(imgs_list[i])
                s = "Real: [ "
                for colorstring in imgs_colors[i]:
                    s+= colorstring + " "
                s+= "] \n" + imgs_shapes[i]
                infoList.append(s)
                if any(cl.lower() == search_string.lower() for cl in imgs_colors[i]):
                    truefalse.append(True)
                else:
                    truefalse.append(False)
        print(truefalse)
        # Uncomment this to show query list in a window
        visualize_retrieval(list, len(list), infoList, truefalse, "Quality: Retrieval by color " + search_string, None)
        return [list, infoList, truefalse]

    def Retrieval_by_shape(imgs_list, labels_knn, search_string, imgs_shapes=None, imgs_colors=None):
        list = []
        infoList = []
        truefalse = []
        for i, label in enumerate(labels_knn):
            if search_string.lower() in label.lower():
                list.append(imgs_list[i])
                s = "Real: [ " + imgs_shapes[i]  + " ] \n"
                for colorstring in imgs_colors[i]:
                    s += colorstring + " "
                infoList.append(s)
                if (imgs_shapes[i].lower() in search_string.lower()):
                    truefalse.append(True)
                else:
                    truefalse.append(False)
        print(truefalse)
        # Uncomment this to show query list in a window
        visualize_retrieval(list, len(list), infoList, truefalse, "Quality: Retrieval by shape " + search_string, None)
        return [list, infoList, truefalse]

    def Retrieval_combined(imgs_list, tags_kmeans, tags_knn, color_string, shape_string, imgs_shapes=None, imgs_colors=None):
        list = []
        infoList = []
        truefalse = []
        for i, (colorLabel, shapeLabel) in enumerate(zip(tags_kmeans, tags_knn)):
            if any(cl.lower() == color_string.lower() for cl in colorLabel) and shape_string.lower() in shapeLabel.lower():
                list.append(imgs_list[i])
                s = "Real: [ " + imgs_shapes[i] + "\n"
                for colorstring in imgs_colors[i]:
                    s += colorstring + " "
                s += " ]"
                infoList.append(s)
                if any(cl.lower() == color_string.lower() for cl in imgs_colors[i]) and (imgs_shapes[i].lower() in shape_string.lower()):
                    truefalse.append(True)
                else:
                    truefalse.append(False)
        # Uncomment this to show query list in a window
        visualize_retrieval(list, len(list), infoList, truefalse, "Quality: Retrieval by color and shape " + color_string + " and " + shape_string, None)
        return [list, infoList]

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
    
    def Kmean_statistics(kmeans, kmax):
        pass
    
    def Get_shape_accuracy(knn_labels, gt_labels):
        matches = 0
        for i, label in enumerate(knn_labels):
            if label == gt_labels[i]:
                matches += 1

        accuracy = (matches/len(knn_labels))*100
        return accuracy

    def Get_color_accuracy(kmeans_labels, gt_labels):
        union_total = 0
        intersection_total = 0
        for i,j in zip(kmeans_labels,gt_labels):
            union = set(i).union(j)
            union_list = list(union)
            union_total += len(union_list)
            intersection = set(i).intersection(j)
            intersection_list = list(intersection)
            intersection_total += len(intersection_list)
        accuracy = (intersection_total/union_total)*100
        return accuracy

    # Getting kmeans labels
    kmeansLabelsList = []

    for img in cropped_images:
        km = KMeans(img)
        km.fit()
        kmeansLabelsList.append(get_colors(km.centroids))

    #Retrieval_by_color(imgs, kmeansLabelsList, "pinK", class_labels, color_labels)

    # Getting knn labels
    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2] * train_imgs.shape[3])
    imgsknn = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2] * imgs.shape[3])
    knn = KNN(train_imgs, train_class_labels)
    knnLabelsList = knn.predict(imgsknn, 3)

    #Retrieval_by_shape(imgs, knnLabelsList, "Handbags", class_labels, color_labels)

    Retrieval_combined(imgs, kmeansLabelsList, knnLabelsList, "red", "haNdbags", class_labels, color_labels)
    
    """
    # Generate KMeans plots

    kmeansLabelsList = []
    kmeansLabelsPlot = []

    for k_value in range(1,10):
        for img in cropped_images:
            km = KMeans(img, k_value)
            km.fit()
            kmeansLabelsList.append(get_colors(km.centroids))
        
        kmeansLabelsPlot.append(Get_color_accuracy(kmeansLabelsList, color_labels))
        kmeansLabelsList.clear()

    fig, ax = plt.subplots()
    ax.set_title("Accuracy of Kmeans")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("K in Kmeans")
    ax.plot(range(1,10), kmeansLabelsPlot, color="tab:purple", marker='o')
    plt.show()
    

    # Generate KNN Plots

    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2] * train_imgs.shape[3])
    imgsknn = imgs.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2] * imgs.shape[3])
    knn = KNN(train_imgs, train_class_labels)
    knnLabelsList = []

    for i in range(3,20):
        knnLabels = (knn.predict(imgsknn, i))
        knnLabelsList.append(Get_shape_accuracy(knnLabels,class_labels))
    
    fig, ax = plt.subplots()
    ax.set_title("Accuracy of KNN")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("k in KNN")
    ax.plot(range(3,20), knnLabelsList, color="tab:purple", marker='o')
    plt.show()
    """
    
    # QUANTITATIVE ANALYSIS TESTS

    # Generate KNN Plots for first tests
    """
    train_imgs = train_imgs.reshape(train_imgs.shape[0], train_imgs.shape[1], train_imgs.shape[2] * train_imgs.shape[3])
    imgsknn = test_imgs.reshape(test_imgs.shape[0], test_imgs.shape[1], test_imgs.shape[2] * test_imgs.shape[3])
    knn = KNN(train_imgs, train_class_labels)
    knnLabelsList = []
    # time_list = []

    for i in range(1,21):
        print(i)
        # start = time.time()

        knnLabels = (knn.predict(imgsknn, i))

        # end = time.time()
        # predicted_time = end - start
        # time_list.append(predicted_time)

        knnLabelsList.append(Get_shape_accuracy(knnLabels,test_class_labels))
    
    fig, ax = plt.subplots()
    ax.set_title("Accuracy of KNN")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("k in KNN")
    ax.plot(range(1,21), knnLabelsList, color="tab:purple", marker='o')
    plt.show()
    """

    # Generate KMeans Plots for first tests
    """
    kmeansLabelsList = []
    kmeansLabelsPlot = []
    # time_list = []

    for k_value in range(2,10):
        print(k_value)
        # start = time.time()

        for img in test_imgs:
            km = KMeans(img, k_value)
            km.fit()
            kmeansLabelsList.append(get_colors(km.centroids))
        
        # end = time.time()
        # predicted_time = end - start
        # time_list.append(predicted_time)

        kmeansLabelsPlot.append(Get_color_accuracy(kmeansLabelsList, test_color_labels))
        kmeansLabelsList.clear()

    fig, ax = plt.subplots()
    ax.set_title("Accuracy of Kmeans")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("K in Kmeans")
    ax.plot(range(2,10), kmeansLabelsPlot, color="tab:purple", marker='o')
    plt.show()
    """
