import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import math

def read_dataset(ROOT_FOLDER = './images/', gt_json='./test/gt.json', w=60, h=80):
    """
        reads the dataset (train and test), returns the images and labels (class and colors) for both sets
    """
    np.random.seed(123)
    ground_truth = json.load(open(gt_json, 'r'))

    train_img_names, train_class_labels, train_color_labels = [], [], []
    test_img_names, test_class_labels, test_color_labels = [], [], []
    for k, v in ground_truth['train'].items():
        train_img_names.append(os.path.join(ROOT_FOLDER, 'train', k))
        train_class_labels.append(v[0])
        train_color_labels.append(v[1])

    for k, v in ground_truth['test'].items():
        test_img_names.append(os.path.join(ROOT_FOLDER, 'test', k))
        test_class_labels.append(v[0])
        test_color_labels.append(v[1])

    train_imgs, test_imgs = load_imgs(train_img_names, test_img_names)

    np.random.seed(42)

    idxs = np.arange(train_imgs.shape[0])
    np.random.shuffle(idxs)
    train_imgs = train_imgs[idxs]
    train_class_labels = np.array(train_class_labels)[idxs]
    train_color_labels = np.array(train_color_labels)[idxs]

    idxs = np.arange(test_imgs.shape[0])
    np.random.shuffle(idxs)
    test_imgs = test_imgs[idxs]
    test_class_labels = np.array(test_class_labels)[idxs]
    test_color_labels = np.array(test_color_labels)[idxs]

    return train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, test_color_labels


def load_imgs(train_img_names, test_img_names, w=60, h=80):
    train_imgs, test_imgs = [], []

    for tr in train_img_names:
        train_imgs.append(read_one_img(tr+ '.jpg'))

    for te in test_img_names:
        test_imgs.append(read_one_img(te + '.jpg'))

    return np.array(train_imgs), np.array(test_imgs)


def read_one_img(img_name, w=60, h=80):
    img = Image.open(img_name)

    img = img.convert("L")

    if img.size != (w, h):
        img = img.resize((w, h))
    return np.array(img)


def visualize_retrieval(imgs, topN, info=None, ok=None, title='', query=None):
    def add_border(color):
        return np.stack([np.pad(imgs[i, :, :, c], 3, mode='constant', constant_values=color[c]) for c in range(3)], axis=2)

    columns = 4
    rows = math.ceil(topN/columns)
    if query is not None:
        fig = plt.figure(figsize=(10, 8*6/8))
        columns += 1
        fig.add_subplot(rows, columns, 1+columns)
        plt.imshow(query)
        plt.axis('off')
        plt.title(f'query', fontsize=8)
    else:
        fig = plt.figure(figsize=(8, 8*6/8))

    for i in range(min(topN, len(imgs))):
        sp = i+1
        if query is not None:
            sp = (sp - 1) // (columns-1) + 1 + sp
        fig.add_subplot(rows, columns, sp)
        if ok is not None:
            im = add_border([0, 255, 0] if ok[i] else [255, 0, 0])
        else:
            im = imgs[i]
        plt.imshow(im)
        plt.axis('off')
        if info is not None:
            plt.title(f'{info[i]}', fontsize=8)
    plt.gcf().suptitle(title)
    plt.show()


# Visualize k-mean with 3D plot
def Plot3DCloud(km, rows=1, cols=1, spl_id=1):
    ax = plt.gcf().add_subplot(rows, cols, spl_id, projection='3d')

    for k in range(km.K):
        Xl = km.X[km.labels == k, :]
        ax.scatter(Xl[:, 0], Xl[:, 1], Xl[:, 2], marker='.', c=km.centroids[np.ones((Xl.shape[0]),dtype='int')*k, :]/255)

    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    ax.set_zlabel('dim 3')
    return ax


def visualize_k_means(kmeans, img_shape):
    def prepare_img(x, img_shape):
        x = np.clip(x.astype('uint8'), 0, 255)
        x = x.reshape(img_shape)
        return x

    fig = plt.figure(figsize=(8, 8))

    X_compressed = kmeans.centroids[kmeans.labels]
    X_compressed = prepare_img(X_compressed, img_shape)

    org_img = prepare_img(kmeans.X, img_shape)

    fig.add_subplot(131)
    plt.imshow(org_img)
    plt.title('original')
    plt.axis('off')

    fig.add_subplot(132)
    plt.imshow(X_compressed)
    plt.axis('off')
    plt.title('kmeans')

    Plot3DCloud(kmeans, 1, 3, 3)
    plt.title('núvol de punts')
    plt.show()


