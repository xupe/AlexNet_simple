import numpy as np
import os
import struct

class mnist(object):
    def __init__(self):
        pass

    def load_mnist(self, path, kind='train'):
        """Load MNIST data from `path`"""
        labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
            self.train_labels = labels

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
            self.train_images = images
        return images, labels

    def random_train_data(self, batch_size, class_num):
        index = np.random.choice(self.train_images.shape[0], batch_size)
        x_batch = self.train_images[index, :]
        y_batch = self.train_labels[index]
        tmp = np.zeros((batch_size, class_num))
        count = 0
        for y in y_batch:
            tmp[count, y] = 1
            count += 1
        return x_batch.astype(np.float32), tmp.astype(np.float32)
