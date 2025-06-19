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

sys.stderr.close()
sys.stderr = stderr_backup

from pixal.modules.model_training import resolve_loss
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

        self.output_layer = Dense(input_dim, activation=self.params['output_activation'], name="output")


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
        loss_fn = resolve_loss(params['loss_function'])
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

        # Compile the model
        self.logger.info("Compiling the model...")
        self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
                    loss=loss_fn,
                    metrics=['mse']) # Mean Squared Error (measures reconstruction quality)
        
        if params['use_gradient_tape']:
            self.logger.info("Training with gradient tape...")

            for epoch in range(params['epochs']):
                for step, (x_batch, y_batch) in enumerate(zip(x_train, y_train)):  # Include labels
                    with tf.GradientTape() as tape:
                        predictions = self.call(x_batch, y_batch)  # Pass both image & label
                        loss = self.compiled_loss(x_batch, predictions)  # MSE loss
                    gradients = tape.gradient(loss, self.trainable_variables)
                    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

                    if step % 10 == 0:
                        self.logger.info(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.numpy():.4f}")
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
            #Args: training data, target data, batch_size, epochs, verbose, callbacks, validation_data=[x_val, x_target], val_batch_size
            # Train the model using (x_train, y_train) as input
            history = self.fit(
                inputs_train,  # Input: (image data + labels)
                x_train,  # Target is the same as input (autoencoder behavior)
                batch_size=params['batchsize'],
                epochs=params['epochs'],
                verbose=1,
                callbacks=[early_stopping, checkpoint],
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
        """Save the model to the specified path."""
        self.logger.info(f"Saving model weights to {save_path}...")
        self.save_weights(save_path)
        self.logger.info(f"Model weights saved to {save_path}")
      
    @classmethod
    def load_model(cls, load_path, params):
        model = cls(params)
        model.build_model(input_dim=params["input_dim"])

        dummy_input = tf.zeros((1, params["input_dim"]))
        if params.get("one_hot_encoding", False):
            dummy_labels = tf.zeros((1, params["label_latent_size"]))
            model([dummy_input, dummy_labels])
        else:
            model(dummy_input)
            
        model.load_weights(load_path)
        return model