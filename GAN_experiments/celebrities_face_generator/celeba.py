# Class for CelebA dataset

import os
import numpy as np
from glob import glob
from PIL import Image

from config import cfg

class CelebA(object):
    def __init__(self):
        files_path = os.path.join(cfg.DATASET_DIR, 'img_align_celeba', '*.jpg')
        self.image_mode = 'RGB'
        self.data_files = glob(files_path)
        self.nb_images = len(self.data_files)
        self.shape = (28, 28, 3)

    def get_image(self, image_path):
        image = Image.open(image_path)
        # Resize image to the good size (28x28)
        j = (image.size[0] - min(image.size)) // 2
        i = (image.size[1] - min(image.size)) // 2
        image = image.crop([j, i, j + min(image.size), i + min(image.size)])
        image = image.resize(self.shape[:2], Image.BILINEAR)
        return np.array(image.convert(self.image_mode))

    def get_batches(self, batch_size):
        # Generate the batches of the dataset, also normalize images to [0,1]
        for i in range(self.nb_images // batch_size):
            img_from = i*batch_size
            img_to = img_from + batch_size
            files = self.data_files[img_from:img_to]
            data_batch = np.array([self.get_image(img) for img in files])
            yield data_batch.astype(np.float32) / 255 - 0.5