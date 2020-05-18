import numpy as np
import tensorflow as tf
from EmbeddingHelper import EmbeddingHelper
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.python.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import os
import argparse


# custom layer used to make VAE
class Sampling(Layer):
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sampling, self).build(input_shape)

    def call(self, inputs, **kwargs):
        z_mean, z_log_sigma = inputs
        epsilon = K.random_normal(shape=tf.shape(z_mean), mean=0., stddev=1)
        return z_mean + K.exp(z_log_sigma) * epsilon

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[0]
        return output_shape


# custom layer used to make VAE
class KLDivergenceLayer(Layer):
    def __init__(self, *args, **kwargs):
        # self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        z_mean, z_log_sigma = inputs

        kl_loss = 1 + 2 * z_log_sigma - K.square(z_mean) - K.exp(2 * z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss = K.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        self.add_loss(kl_loss)
        return inputs


# data generator created using tf data
def data_generator(size=10000, batch_size=32):
    (x_train, y_train), (x_test, _) = load_data()
    x_train = x_train[0:size, ...]
    x_train = x_train[..., np.newaxis]

    x_train = x_train.astype('float32') / 255
    y_train = y_train[0:size, ...]
    train_size = size

    img_ds = tf.data.Dataset.from_tensor_slices(x_train)
    label_ds = tf.data.Dataset.from_tensor_slices(y_train)
    # The generator needs to output both data and the corresponding label.
    # If label is not present (for unsupervised tasks), use a placeholder to make sure it outputs something for the label
    embedding_ds = tf.data.Dataset.zip((img_ds, label_ds))
    embedding_ds = embedding_ds.shuffle(train_size)
    embedding_ds = embedding_ds.batch(batch_size=batch_size)
    embedding_ds = embedding_ds.prefetch(1)
    return embedding_ds


def load_keras_model(saved_model_path, custom_objects={}):
    # add custom layer to custom_objects
    return load_model(
        saved_model_path,
        custom_objects=custom_objects
    )


def extract_encoder_from_vae(saved_model_path, custom_objects):
    vae = load_keras_model(saved_model_path=saved_model_path, custom_objects=custom_objects)
    _encoder = vae.get_layer("encoder")
    sampling = vae.get_layer("sampling")(_encoder.output)
    encoder = Model(_encoder.input, sampling)
    return encoder


def main(embedding_dir, saved_model_path):
    os.makedirs(embedding_dir, exist_ok=True)
    custom_objects = {'KLDivergenceLayer': KLDivergenceLayer, "Sampling": Sampling}
    encoder = extract_encoder_from_vae(saved_model_path=saved_model_path, custom_objects=custom_objects)
    generator = data_generator(size=2000, batch_size=32)

    # Instantiate the enbedding helper
    embedding_helper = EmbeddingHelper(encoder=encoder, data_generator=generator, embeddings_dir=embedding_dir)
    embedding_helper.create_sprite()
    embedding_helper.to_tensorboard()

    label_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple', 'orange', 'chartreuse']
    embedding_helper.tsne_plot(labels=label_list, colors=colors, show=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding_dir", type=str, default="embeddings/",
                    help="directory to save the embedding related data")
    ap.add_argument("--saved_model_path", type=str, default="model.hdf5", help="file of trained network")
    args = vars(ap.parse_args())
    main(embedding_dir=args["embedding_dir"], saved_model_path=args["saved_model_path"])
