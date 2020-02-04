import tensorflow as tf
from tensorflow.python.keras import layers


class CyclicalAnnealingSchedule(tf.keras.callbacks.Callback):

    def __init__(self, cycle_duration, summary_writer=None):
        self.cycle_duration = cycle_duration
        self.summary_writer = summary_writer
        self.processed_instances = 0
        self.instances_offset = 0
        self.absolute_batch = 0
        self.batch_offset = 0
        self.kl_divergence_layer = None

    def on_train_begin(self, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, KLDivergence):
                self.kl_divergence_layer = layer
                print("KL divergence layer found: " + str(self.kl_divergence_layer.kl_loss_weight))

        if self.kl_divergence_layer is None:
            raise Exception("No KL divergence layer found in current model")

    def on_train_batch_end(self, batch, logs=None):
        self.processed_instances += logs['size']
        self.absolute_batch = batch + self.batch_offset
        state = (self.processed_instances - self.instances_offset) % self.cycle_duration

        if state <= self.cycle_duration / 3:
            self.model.layers[3].kl_loss_weight.assign(0)

        elif self.cycle_duration/3 <= state <= 2*self.cycle_duration/3:
            self.model.layers[3].kl_loss_weight.assign((3 / self.cycle_duration) * (state - self.cycle_duration/3))

        elif self.cycle_duration >= state >= 2*self.cycle_duration/3:
            self.model.layers[3].kl_loss_weight.assign(1)

        elif state > self.cycle_duration:  # Restart cycle
            self.instances_offset = self.processed_instances

        if self.summary_writer:
            with self.summary_writer.as_default():
                tf.summary.scalar('kl_loss_weight', self.model.layers[3].kl_loss_weight, step=self.absolute_batch)

    def on_epoch_end(self, epoch, logs=None):
        self.batch_offset = self.absolute_batch


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, using the reparametrization trick to allow back propagation."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(layers.Layer):
    """Maps Kinematic Joint Positions to a triplet (z_mean, z_log_var, z)."""

    def __init__(self,
                 latent_dim=32,
                 intermediate_dim=64,
                 name='Encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu', name="encoder_hidden_layer")
        self.dense_mean = layers.Dense(latent_dim, name="mean_layer")
        self.dense_log_var = layers.Dense(latent_dim, name="log_var_layer")
        self.sampling = Sampling(name="Gaussian_Sampling")

    def call(self, inputs):
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self,
                 original_dim,
                 intermediate_dim=64,
                 name='Decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = layers.Dense(intermediate_dim, activation='relu', name="decoder_hidden_layer")
        # Output layer without activation to reproduce kinematic angle data
        self.dense_output = layers.Dense(original_dim, name="output_layer")

    def call(self, inputs):
        # Inputs are the
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class KLDivergence(layers.Layer):
    """Converts z, the encoded digit vector, back into a readable digit."""

    def __init__(self, initial_weight=0.0,
                 name='KL-Divergence',
                 **kwargs):
        super(KLDivergence, self).__init__(name=name, **kwargs)

        self.kl_loss_weight = tf.Variable(initial_weight, trainable=False, name="kl_loss_weight", dtype=tf.float32)

    def call(self, inputs):
        z_mean, z_log_var, z = inputs
        # Add KL divergence regularization loss for this forward pass
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        return self.kl_loss_weight * kl_loss

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'kl_loss_weight': self.kl_loss_weight,
        })
        return config


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(self,
                 original_dim=18,
                 intermediate_dim=64,
                 latent_dim=2,
                 name='Autoencoder',
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,
                               intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim=original_dim,
                               intermediate_dim=intermediate_dim)
        self.kl_loss_weight = 0
        self.step = 0

    def call(self, inputs):
        # Forward pass of the Encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        # Forward pass of the Decoder taken the re-parameterized z latent variable
        reconstructed = self.decoder(z)

        # Add KL divergence regularization loss for this forward pass
        kl_loss = - 0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

        self.add_loss(self.kl_loss_weight * kl_loss)

        return reconstructed
