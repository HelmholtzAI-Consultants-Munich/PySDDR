import numpy as np
import os
import imageio

mnist = np.load('mnist.npy')
num_imgs = mnist.shape[0]
for i in range(num_imgs):
    img = mnist[i,:,:]
    name = 'img_%s.jpg'%(i)
    file_path = os.path.join('mnist_images', name)
    imageio.imwrite(file_path, (img*255).astype(np.uint8))