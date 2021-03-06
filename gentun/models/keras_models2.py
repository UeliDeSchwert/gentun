#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using Keras
"""

import tensorflow.keras.backend as K
import numpy as np

from keras.layers import Input, LSTM, Activation, Add, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import StratifiedKFold, KFold

from .generic_models import GentunModel

K.set_image_data_format('channels_last')


class GeneticLSTMModel(GentunModel):

    def __init__(self, x_train, y_train, genes, nodes, input_shape, kernels_per_layer, dense_units,
                 dropout_probability, last_lstm_units,
                 kfold=5, epochs=(3,), learning_rate=(1e-3,), batch_size=32):
        super(GeneticLSTMModel, self).__init__(x_train, y_train)
        self.model = self.build_model(
            genes, nodes, input_shape, kernels_per_layer,
            dense_units, dropout_probability, last_lstm_units
        )
        self.name = '-'.join(gene for gene in genes.values())
        self.kfold = kfold
        if type(epochs) is int and type(learning_rate) is int:
            self.epochs = (epochs,)
            self.learning_rate = (learning_rate,)
        elif type(epochs) is tuple and type(learning_rate) is tuple:
            self.epochs = epochs
            self.learning_rate = learning_rate
        else:
            print(epochs, learning_rate)
            raise ValueError("epochs and learning_rate must be both either integers or tuples of integers.")
        self.batch_size = batch_size

    def plot(self):
        """Draw model to validate gene-to-DAG."""
        from keras.utils import plot_model
        plot_model(self.model, to_file='{}.png'.format(self.name))

    @staticmethod
    def build_dag(x, nodes, connections, kernels):
        # Get number of nodes (K_s) using the fact that K_s*(K_s-1)/2 == #bits
        # nodes = int((1 + (1 + 8 * len(connections)) ** 0.5) / 2)
        # Separate bits by whose input they represent (GeneticCNN paper uses a dash)
        ctr = 0
        idx = 0
        separated_connections = []
        while idx + ctr < len(connections):
            ctr += 1
            separated_connections.append(connections[idx:idx + ctr])
            idx += ctr
        # Get outputs by node (dummy output ignored)
        outputs = []
        for node in range(nodes - 1):
            node_outputs = []
            for i, node_connections in enumerate(separated_connections[node:]):
                if node_connections[node] == '1':
                    node_outputs.append(node + i + 1)
            outputs.append(node_outputs)
        outputs.append([])
        # Get inputs by node (dummy input, x, ignored)
        inputs = [[]]
        for node in range(1, nodes):
            node_inputs = []
            for i, connection in enumerate(separated_connections[node - 1]):
                if connection == '1':
                    node_inputs.append(i)
            inputs.append(node_inputs)
        # Build DAG
        output_vars = []
        all_vars = [None] * nodes
        for i, (ins, outs) in enumerate(zip(inputs, outputs)):
            if ins or outs:
                if not ins:
                    tmp = x
                else:
                    add_vars = [all_vars[i] for i in ins]
                    if len(add_vars) > 1:
                        tmp = Add()(add_vars)
                    else:
                        tmp = add_vars[0]
                tmp = LSTM(kernels, activation="relu", return_sequences=True)(tmp)
                all_vars[i] = tmp
                if not outs:
                    output_vars.append(tmp)
        if len(output_vars) > 1:
            return Add()(output_vars)
        return output_vars[0]

    def build_model(self, genes, nodes, input_shape, units_per_layer, dense_units,
                    dropout_probability, last_lstm_units):
        x_input = Input(input_shape)
        x = x_input
        for layer, units in enumerate(units_per_layer):
            # Default input node
            x = LSTM(units, activation="relu", return_sequences=True)(x)
            x = Dropout(dropout_probability)(x)
            # Decode internal connections
            connections = genes['S_{}'.format(layer + 1)]
            # If at least one bit is 1, then we need to construct the Directed Acyclic Graph
            if not all([not bool(int(connection)) for connection in connections]):
                x = self.build_dag(x, nodes[layer], connections, units)
                # Output node
                x = LSTM(units, activation="relu", return_sequences=True)(x)
        x = LSTM(last_lstm_units, activation="relu")(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout_probability)(x)
        x = Dense(1, activation='linear')(x)
        return Model(inputs=x_input, outputs=x, name='GeneticLSTM')

    def reset_weights(self):
        """Initialize model weights."""
        pass

    def cross_validate(self):
        """Train model using k-fold cross validation and
        return mean value of the validation accuracy.
        """
        acc = .0
        kfold = KFold(n_splits=self.kfold, shuffle=True)
        for fold, (train, validation) in enumerate(kfold.split(self.x_train, self.y_train)):
            print("KFold {}/{}".format(fold + 1, self.kfold))
            self.reset_weights()
            for epochs, learning_rate in zip(self.epochs, self.learning_rate):
                print("Training {} epochs with learning rate {}".format(epochs, learning_rate))
                self.model.compile(optimizer=Adam(lr=learning_rate), loss='mae', metrics=['mse', 'mape'])
                self.model.fit(
                    self.x_train[train], self.y_train[train], epochs=epochs, batch_size=self.batch_size, verbose=1
                )
            acc += self.model.evaluate(self.x_train[validation], self.y_train[validation], verbose=0)[1] / self.kfold
        return acc
