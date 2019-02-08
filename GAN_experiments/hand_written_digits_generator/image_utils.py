# Converts a flattened image into its 2-dimensional representation
def images_to_vectors(images):
    return images.view(images.size(0), 784)

# Converts 2 dimensional image to flattened one
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)