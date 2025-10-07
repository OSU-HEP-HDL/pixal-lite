import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')

stderr_backup = sys.stderr
sys.stderr = open(os.devnull, 'w')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import initializers


import tensorflow as tf


# Only after TF has initialized should you import Numba/onnxruntime (if you must).
# from numba import cuda  # <-- avoid calling cuda.select_device(0) before TF
# import onnxruntime as ort

sys.stderr.close()
sys.stderr = stderr_backup

from pixal.modules.model_training import  make_weighted_loss,make_per_channel_metric, make_total_weighted_metric, make_weighted_contrib_metric
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
import json



class Autoencoder(tf.keras.Model):

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)
        self.params = params
        self.encoder = None
        self.decoder = None
        self.output_layer = None
        self.latent_layer = None
        self.label_projection = None
        self.one_hot_encoding = params.get("one_hot_encoding", False)
        self.logger = logging.getLogger("pixal")


    def call(self, inputs):
        self.logger.debug("Starting forward pass...")
        
        if self.one_hot_encoding:
            x, labels = inputs
        else:
            x = inputs
        
        # 🔹 Encode
        encoded = self.encoder(x)
        latent = self.latent_layer(encoded)

        # 🔹 Decode
        if self.one_hot_encoding:
            transformed_labels = self.label_projection(labels)
            latent_combined = tf.concat([latent, transformed_labels], axis=1)
        else:
            latent_combined = latent

        decoded = self.decoder(latent_combined)
        self.logger.debug("Forward pass complete.")
        return self.output_layer(decoded)
        

    def build_model(self, input_dim):
        self.logger.info(f"Building model with input_dim={input_dim}")
        
        self.logger.info("Initializing Autoencoder model...")
        self.logger.info(f"Autoencoder model architecture: {self.params['architecture']}")

        latent_dim = self.params['architecture'][-1]
        reg_type = self.params.get('regularization')
        
        if reg_type == 'l1':
            regularizer = tf.keras.regularizers.l1(self.params.get('l1_regularization', 0.001))
        elif reg_type == 'l2':
            regularizer = tf.keras.regularizers.l2(self.params.get('l2_regularization', 0.001))
        elif reg_type == 'l1_l2':
            regularizer = tf.keras.regularizers.l1_l2(
                l1=self.params.get('l1_regularization', 0.001),
                l2=self.params.get('l2_regularization', 0.001)
            )
        else:
            regularizer = None

        # Build encoder
        self.encoder = tf.keras.Sequential(name="encoder")
        self.encoder.add(Input(shape=(input_dim,)))
        for i, units in enumerate(self.params['architecture'][:-1]):
            self.encoder.add(Dense(units, activation=tf.nn.leaky_relu, activity_regularizer=regularizer,
                                name=self.params['encoder_names'][i]))

        self.latent_layer = Dense(latent_dim, activation=tf.nn.leaky_relu, name="latent")
        
        if self.one_hot_encoding:
            self.label_projection = Dense(self.params['label_latent_size'], activation='relu', name="label_transform")

        # Build decoder
        decoder_arch = self.params['architecture'][:-1][::-1]
        decoder_input_dim = latent_dim + self.params['label_latent_size'] if self.one_hot_encoding else latent_dim

        self.decoder = tf.keras.Sequential(name="decoder")
        self.decoder.add(Input(shape=(decoder_input_dim,)))
        for i, units in enumerate(decoder_arch):
            self.decoder.add(Dense(units, activation=tf.nn.leaky_relu, activity_regularizer=regularizer,
                                name=self.params['decoder_names'][i]))
        
        if self.params.get('bias_init_vector') is not None:
            bias_init = initializers.Constant(self.params['bias_init_vector'])
        else:
            bias_init = None
        
        self.output_layer = Dense(input_dim, 
                                activation=self.params['output_activation'], 
                                name="output",
                                bias_initializer=bias_init,
                                kernel_regularizer=None,
                                activity_regularizer=None)


    def get_config(self):
        config = super().get_config()
        config.update({
            'params': self.params
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        params = config.pop('params')
        return cls(params, **config)

    def compile_and_train(self, x_train, y_train, x_val, y_val, params):
        # Early stopping and checkpointing

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=params['patience'],
            mode='min',
            verbose=1,
            restore_best_weights=True
        )

        checkpoint = ModelCheckpoint(
            filepath=f"{params['model_path']}/{params['modelName']}.keras", 
            verbose=1, 
            save_freq='epoch'
        )

        channels = params.get('channels', [])

        if params['optimizer'] == 'adam':
           optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['learning_rate']))
        elif params['optimizer'] == 'adamW':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=float(params['learning_rate']), weight_decay=float(params['weight_decay']))
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=float(params['learning_rate']))
        # Build metrics and loss function

        metrics = ["mse"] + [make_per_channel_metric(i, channels, reducer="mse") for i in range(len(channels))]
        metrics.append(make_total_weighted_metric(channels, params['weights'], base="huber", delta=1.0))
        metrics += [make_weighted_contrib_metric(i, channels, params['weights'], base="huber", delta=0.5)
            for i in range(len(channels))]
        
        loss_fn = make_weighted_loss(channels, weights=params['weights'], base=params['loss_function'], delta=params.get('huber_delta', 1.0),from_logits=params.get('logits',False) , use_mask=params.get('masked_loss', False))

        # Compile the model
        self.logger.info("Compiling the model...")
        self.compile(optimizer=optimizer,
                    loss=loss_fn,
                    metrics=metrics,
                    run_eagerly=True) 
        
        if params['use_gradient_tape']:
            self.logger.info("Training with gradient tape...")
            for epoch in range(params['epochs']):
                epoch_loss = 0.0
                epoch_steps = 0
                for step, (x_batch, y_batch) in enumerate(zip(x_train, y_train)):  # Include labels
                    with tf.GradientTape() as tape:
                        predictions = self.call(x_batch, y_batch)  # Pass both image & label
                        loss = self.compiled_loss(x_batch, predictions)  # MSE loss
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                    epoch_loss += float(loss.numpy())
                    epoch_steps += 1

                    if step % 10 == 0:
                        self.logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.numpy():.4f}")

                # end epoch: log average epoch loss to MLflow if available
                try:
                    from pixal.mlflow_utils import log_metrics
                    avg_loss = epoch_loss / max(1, epoch_steps)
                    log_metrics({"loss": float(avg_loss)}, step=epoch)
                except Exception:
                    pass
        else:
            if self.one_hot_encoding:
                inputs_train = [x_train, y_train]
                inputs_val = [x_val, y_val]
                val_targets = x_val
            else:
                inputs_train = x_train
                inputs_val = x_val
                val_targets = x_val
            # Train the model with default fit method
            self.logger.info("Training the model with standard fit method...")
            # Prepare callbacks and add MLflow Keras callback when available
            callbacks = [early_stopping, checkpoint]
            try:
                from pixal.mlflow_utils import KerasModelLoggerCallback
                # Only append if the callback is a real TF callback
                if KerasModelLoggerCallback is not None:
                    callbacks.append(KerasModelLoggerCallback())
            except Exception:
                pass

            # Args: training data, target data, batch_size, epochs, verbose, callbacks, validation_data=[x_val, x_target], val_batch_size
            # Train the model using (x_train, y_train) as input
            history = self.fit(
                inputs_train,  # Input: (image data + labels)
                x_train,  # Target is the same as input (autoencoder behavior)
                batch_size=params['batchsize'],
                epochs=params['epochs'],
                verbose=1,
                callbacks=callbacks,
                validation_data=(inputs_val, val_targets),
                validation_batch_size=params['batchsize']
        ) 
        # Save loss values to a file
        loss_history = {
            "train_loss": history.history["loss"],
            "val_loss": history.history["val_loss"]
            }
        save_dir = params['model_path']
        self.logger.info("save_dir: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)  # Creates the directory if it does not exist
        loss_file_path = os.path.join(save_dir, "loss_history.json")
        with open(loss_file_path, "w") as f:
            json.dump(loss_history, f)

        # Plot and save the loss curve
        self.plot_loss(history.history, params['fig_path'])

    def plot_loss(self, history, save_dir):
        """Plot and save training vs validation loss"""
        plt.figure(figsize=(8, 5))
        plt.plot(history['loss'], label="Train Loss")
        plt.plot(history['val_loss'], label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid()
        
        loss_plot_path = os.path.join(save_dir, "loss_plot.png")
        plt.savefig(loss_plot_path)
        plt.close()
        
        self.logger.info(f"Loss plot saved to {loss_plot_path}")

    def evaluate_model(self, test_data, test_labels=None):
        if self.one_hot_encoding:
            return self.evaluate([test_data, test_labels], test_data)
        else:
            return self.evaluate(test_data, test_data)

    def predict_model(self, new_data,labels=None):
        if self.one_hot_encoding:
            return self.predict([new_data, labels])
        else:
            return self.predict(new_data)
    
    def save_model(self, save_path):
        """Save the model.

        The training scripts pass a `save_path` derived from the configured
        `model_name` and `model_file_extension`. To be robust we attempt to
        save the full Keras model (recommended) and also write weights to a
        backup `<model_name>.weights.h5` for backward compatibility.
        """
        try:
            # Attempt to save the full model (SavedModel or single-file format)
            self.logger.info(f"Saving full Keras model to {save_path}...")
            # tf.keras.Model.save will create a directory for SavedModel or a file for .h5
            self.save(save_path)
            self.logger.info(f"Full model saved to {save_path}")
        except Exception as e:
            self.logger.warning(f"Full model save failed ({e}), falling back to weights-only save.")

        # Always save weights as a backup (HDF5)
        weights_path = os.path.splitext(str(save_path))[0] + ".weights.h5"
        try:
            self.logger.info(f"Saving model weights to {weights_path}...")
            self.save_weights(weights_path)
            self.logger.info(f"Model weights saved to {weights_path}")
        except Exception as e:
            self.logger.error(f"Failed to save weights to {weights_path}: {e}")
      
    @classmethod
    def load_model(cls, load_path, params):
        """Load a model with preference order:

        1. If `load_path` points to a saved full model (SavedModel dir or HDF5), use `tf.keras.models.load_model`.
        2. Otherwise, rebuild the model architecture and call `load_weights` from a weights file.

        The `params` dict is used to reconstruct the model when needed.
        """
        # First try to load a full model
        try:
            loaded = tf.keras.models.load_model(load_path)
            # If successful, ensure the returned object is an Autoencoder instance
            if isinstance(loaded, cls):
                return loaded
            else:
                # Wrap weights into our Autoencoder class: create and load weights from the loaded model
                model = cls(params)
                model.build_model(input_dim=params["input_dim"])
                # try to load weights from the loaded model if possible
                try:
                    model.set_weights(loaded.get_weights())
                    return model
                except Exception:
                    # fallthrough to weights file strategy
                    pass
        except Exception:
            # Not a full-model file or load failed; continue to weights fallback
            pass

        # Fallback: rebuild architecture and load weights
        model = cls(params)
        model.build_model(input_dim=params["input_dim"])

        dummy_input = tf.zeros((1, params["input_dim"]))
        if params.get("one_hot_encoding", False):
            dummy_labels = tf.zeros((1, params["label_latent_size"]))
            # run a forward pass to build layers
            model([dummy_input, dummy_labels])
        else:
            model(dummy_input)

        # If provided load_path points to a weights file use it, otherwise try a derived .weights.h5
        try:
            model.load_weights(load_path)
            return model
        except Exception:
            weights_path = os.path.splitext(str(load_path))[0] + ".weights.h5"
            model.load_weights(weights_path)
            return model