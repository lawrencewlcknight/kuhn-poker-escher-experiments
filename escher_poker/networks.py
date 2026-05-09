"""TensorFlow/Keras networks used by the ESCHER solver."""

from __future__ import annotations

import tensorflow as tf


class SkipDense(tf.keras.layers.Layer):
    """Dense layer with a residual connection."""

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.hidden = tf.keras.layers.Dense(units, kernel_initializer="he_normal")

    def call(self, x):
        return self.hidden(x) + x


class PolicyNetwork(tf.keras.Model):
    """Average-policy MLP with optional residual hidden layers."""

    def __init__(self, input_size, policy_network_layers, num_actions, activation="leakyrelu", **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        if activation == "leakyrelu":
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == "relu":
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.softmax = tf.keras.layers.Softmax()
        self.hidden = []
        prevunits = 0
        for units in policy_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(tf.keras.layers.Dense(units, kernel_initializer="he_normal"))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(policy_network_layers[-1], kernel_initializer="he_normal")
        self.out_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)
        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        x = tf.where(mask == 1, x, -10e20)
        return self.softmax(x)


class RegretNetwork(tf.keras.Model):
    """Regret/advantage approximation network."""

    def __init__(self, input_size, regret_network_layers, num_actions, activation="leakyrelu", **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        self._num_actions = num_actions
        if activation == "leakyrelu":
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == "relu":
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.hidden = []
        prevunits = 0
        for units in regret_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(tf.keras.layers.Dense(units, kernel_initializer="he_normal"))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(regret_network_layers[-1], kernel_initializer="he_normal")
        self.out_layer = tf.keras.layers.Dense(num_actions)

    @tf.function
    def call(self, inputs):
        x, mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)
        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        x = self.out_layer(x)
        return mask * x


class ValueNetwork(tf.keras.Model):
    """History-value network used by ESCHER to estimate regret targets."""

    def __init__(self, input_size, val_network_layers, activation="leakyrelu", **kwargs):
        super().__init__(**kwargs)
        self._input_size = input_size
        if activation == "leakyrelu":
            self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
        elif activation == "relu":
            self.activation = tf.keras.layers.ReLU()
        else:
            self.activation = activation

        self.hidden = []
        prevunits = 0
        for units in val_network_layers[:-1]:
            if prevunits == units:
                self.hidden.append(SkipDense(units))
            else:
                self.hidden.append(tf.keras.layers.Dense(units, kernel_initializer="he_normal"))
            prevunits = units
        self.normalization = tf.keras.layers.LayerNormalization()
        self.lastlayer = tf.keras.layers.Dense(val_network_layers[-1], kernel_initializer="he_normal")
        self.out_layer = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        x, _mask = inputs
        for layer in self.hidden:
            x = layer(x)
            x = self.activation(x)
        x = self.normalization(x)
        x = self.lastlayer(x)
        x = self.activation(x)
        return self.out_layer(x)
