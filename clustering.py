from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans


def hierarchical_clustering(similarity_matrix):
    distance_matrix = 1 - np.array(similarity_matrix)
    Z = linkage(distance_matrix, 'ward')
    clusters = fcluster(Z, t=1.15, criterion='distance')
    return clusters


def vae_clustering(fingerprints_binary):
    # Debugging statement to check the structure of fingerprints
    print("Fingerprints shape:", fingerprints_binary.shape)

    vae, encoder = train_vae(fingerprints_binary)
    encoded_fingerprints = encoder.predict(fingerprints_binary)
    kmeans = KMeans(n_clusters=10)
    clusters = kmeans.fit_predict(encoded_fingerprints)
    return clusters


class CustomVariationalLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, inputs, outputs, z_mean, z_log_var):
        reconstruction_loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
        reconstruction_loss *= inputs.shape[1]
        kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        kl_loss = tf.reduce_sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.reduce_mean(reconstruction_loss + kl_loss)

    def call(self, inputs, outputs, z_mean, z_log_var):
        loss = self.vae_loss(inputs, outputs, z_mean, z_log_var)
        self.add_loss(loss)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1]


def train_vae(fingerprints_binary):
    input_dim = fingerprints_binary.shape[1]
    latent_dim = 2  # Dimension of latent space

    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    h = layers.Dense(128, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_h = layers.Dense(128, activation='relu')
    decoder_mean = layers.Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # VAE model
    outputs = CustomVariationalLayer()(inputs, x_decoded_mean, z_mean, z_log_var)
    vae = models.Model(inputs, outputs)

    vae.compile(optimizer='rmsprop')
    vae.fit(fingerprints_binary, fingerprints_binary, epochs=50, batch_size=16, validation_split=0.2)

    # Encoder model
    encoder = models.Model(inputs, z_mean)
    return vae, encoder
