import numpy as np
from IPython.display import HTML

#%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torchvision.utils as vutils

def plot_losses(losses):
    plt.figure(figsize=(10,5))
    plt.title("Losses during training of {}".format(
              ', '.join([l['name'] for l in losses])))
    for loss in losses:
        plt.plot(loss['loss'], label=loss['name'])
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def generate_progression(img_list):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

def display_comparaison(real_batch, img_list, device):
    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()
