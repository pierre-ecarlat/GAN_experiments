#!/usr/bin/env python3

import os
import cv2
import numpy as np
from glob import glob
import tensorflow as tf

from config import cfg
from celeba import CelebA
from generators import generator_tf as G
from discriminators import discriminator_tf as D


def model_inputs(w, h, c, z_dim):
    # Input for the model, image
    _i = tf.placeholder(tf.float32, [None, w, h, c], 'real_input_images')
    _z = tf.placeholder(tf.float32, [None, z_dim], 'input_z')
    return _i, _z

def model_loss(inp_real, inp_z, out_channel, alpha=0.2, smooth_factor=0.1):
    # Loss from the real image for G and D
    d_model_real, d_logits_real = D(inp_real, alpha=alpha)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_real,
            labels=tf.ones_like(d_model_real) * (1 - smooth_factor)))

    # Loss from the fake image for G and D
    inp_fake = G(inp_z, out_channel, alpha=alpha)
    d_model_fake, d_logits_fake = D(inp_fake, reuse=True, alpha=alpha)
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake,
            labels=tf.zeros_like(d_model_fake)))

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=d_logits_fake,
            labels=tf.ones_like(d_model_fake)))

    return d_loss_real + d_loss_fake, g_loss

def model_opt(d_loss, g_loss, lr, beta1):
    # Optimization operations for the losses of D and G
    # Get weights and bias to update
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(lr, beta1=beta1)\
                              .minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(lr, beta1=beta1)\
                              .minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

def show_generator_output(sess, n_img, inp_z, out_channel, img_mode='RGB'):
    # Show the output from the generator
    cmap = None if img_mode == 'RGB' else 'gray'
    z_dim = inp_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_img, z_dim])

    samples = sess.run(G(inp_z, out_channel, False), feed_dict={inp_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    return images_grid

def train(nb_epochs, batch_size, z_dim, lr, beta1, get_batches, data_shape,
          data_img_mode, print_every=10, show_every=100):
    # Train method for the GAN
    inp_real, inp_z = model_inputs(*data_shape, z_dim)
    d_loss, g_loss = model_loss(inp_real, inp_z, data_shape[2])
    d_train_opt, g_train_opt = model_opt(d_loss, g_loss, lr, beta1)

    saver = tf.train.Saver()
    sample_z = np.random.uniform(-1, 1, size=(72, z_dim))

    samples, losses = [], []

    steps = 0
    count = 0
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # continue training
        save_path = saver.save(sess, "/tmp/model.ckpt")
        ckpt = tf.train.latest_checkpoint('./model/')
        saver.restore(sess, save_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        os.mkdir('output')
        for epoch_i in range(nb_epochs):
            os.mkdir('output/'+ str(epoch_i))
            for batch_images in get_batches(batch_size):
                steps += 1
                batch_images *= 2.0
                
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                # Run optimizers
                sess.run(d_train_opt, feed_dict={inp_real: batch_images, input_z: batch_z})
                sess.run(g_train_opt, feed_dict={input_z: batch_z})
                
                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({inp_real: batch_images, inp_z: batch_z})
                    train_loss_g = g_loss.eval({inp_z: batch_z})
                    print("Epoch {}/{} Step {}...".format(epoch_i+1, nb_epochs, steps),
                      "Discriminator Loss: {:.4f}...".format(train_loss_d),
                      "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses for viewing after training
                    losses.append((train_loss_d, train_loss_g))

                if steps % show_every == 0:
                    count = count +1
                    iterr = count*show_every
                    # Show example output for the generator
                    images_grid = show_generator_output(sess, 25, inp_z, data_shape[2], data_img_mode)
                    dst = os.path.join("output", str(epoch_i), str(iterr)+".png")
                    pyplot.imsave(dst, images_grid)
                    
                 # saving the model
                if epoch_i % 10 == 0:
                    if not os.path.exists('./model/'):
                        os.makedirs('./model')
                    saver.save(sess, './model/' + str(epoch_i))

# Get the data in a readable format
dataset = CelebA()

# Tensorflow
with tf.Graph().as_default():
    train(cfg.NB_EPOCHS, cfg.BATCH_SIZE, cfg.SIZE_G_INPUT, cfg.LEARNING_RATE,
          cfg.BETA1, dataset.get_batches, dataset.shape, dataset.image_mode)


for f in glob("output/**/*.png"):
    image = cv2.imread(f)
    # cv2.imshow('my_image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    large = cv2.resize(image, (0,0), fx=3, fy=3)
    cv2.imwrite(f, large)