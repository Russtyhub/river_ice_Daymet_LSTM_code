#!/usr/bin/python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import numpy as np
import tensorflow as tf
import keras_tuner as kt

class LSTM(kt.HyperModel):

    def __init__(self,
                 input_shape,
                 # batch_size,
                 # gpus_per_node,
                 optimizer,
                 loss,
                 max_number_lstm_layers,
                 max_number_dense_layers,
                 strategy = None,
                 metrics = None,
                 momentum=None,
                 initial_bias=None,
                 mask_value = -1.0,
                 ):

        self.input_shape = input_shape
        # self.gpus_per_node = gpus_per_node
        # self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.max_number_lstm_layers = max_number_lstm_layers
        self.max_number_dense_layers = max_number_dense_layers
        self.strategy = strategy
        self.metrics = metrics
        self.momentum = momentum
        self.initial_bias = initial_bias
        self.mask_value = mask_value

    def convert_loss(self):

        # Regression:
        if self.loss.upper() == 'MSLE':
            loss_fn = tf.keras.losses.MeanSquaredLogarithmicError(name="msle")
        elif self.loss.upper() == 'MSE':
            loss_fn = tf.keras.losses.MeanSquaredError(name='mse')
        elif self.loss.upper() == 'HUBER':
            loss_fn = tf.keras.losses.Huber(delta=1.0, name='huber')
        elif (self.loss.upper() == 'MAE') or (self.loss.upper() == 'LOSS'):
            loss_fn = tf.keras.losses.MeanAbsoluteError(name='mae')

        # Classification
        elif self.loss.upper() == 'BCE':
            loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False, 
                                                         name = 'binary_cross_entropy')
        elif self.loss.upper() == 'SCCE':
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, 
                                                         name = 'sparse_categorical_cross_entropy')
        elif self.loss.upper() == 'CCE':
            loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False, 
                                                         name = 'categorical_cross_entropy')
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
        return loss_fn

    def produce_direction_build(self):
        
        if self.metrics == None:
            print('NO METRICS WERE SPECIFIED')

        else:
            min_metrics = set(['MSLE', 'RMSE', 'MAE', 'MSE', 'LOG_COSH_ERROR', 'COS_SIM'])
            max_metrics = set(['AUC', 'ACCURACY', 'PRECISION', 'RECALL', 'F1SCORE'])
            metric_names = set([metric.name.upper() for metric in self.metrics])

            if metric_names.issubset(min_metrics):
                direction = 'min'
            elif metric_names.issubset(max_metrics):
                direction = 'max'
            else:
                direction = None
                print('CHECK YOUR METRICS!')
                print('COULD BE: SPELLED DIFFERENTLY; UNAVAILABLE; MIXED BETWEEN MIN AND MAX')
                
        return direction

    def produce_direction(self):
        
        if self.strategy:
            with self.strategy.scope():
                direction = self.produce_direction_build()
        else:
            direction = self.produce_direction_build()
        return direction
        
    def build_model_body(self, hp, parameters=None):
        
        param_keys = ['dropout_rate', 'recurrent_dropout_rate', 
              'learning_rate', 'number_of_lstm_layers', 'number_of_dense_layers',
               'batch_size', 'dense_layers_activation_funcs']
        
        if not parameters:
            
            self.parameters_included = False
                        
            parameters = {'dropout_rate' : hp.Float('dropout_rate', 
                                                    min_value=0, 
                                                    max_value=0.8, 
                                                    step = 0.05),
                          'recurrent_dropout_rate' : hp.Float('recurrent_dropout_rate', 
                                                    min_value=0, 
                                                    max_value=0.8, 
                                                    step=0.05),
                          'learning_rate' : hp.Float('learning_rate', 
                                                    min_value=1e-3, 
                                                    max_value=1e-1, 
                                                    sampling='log'),
                          'number_of_lstm_layers' : hp.Int("number_of_lstm_layers", 
                                                    1, 
                                                    self.max_number_lstm_layers),
                          'number_of_dense_layers' : hp.Int("num_dense_layers", 
                                                    1, 
                                                    self.max_number_dense_layers),
                          'batch_size' : hp.Int("batch_size", 64, 1344, 128),
                          'dense_layers_activation_funcs' : hp.Choice("activation", ['elu', 'relu', 'tanh']),
                         }
            
            for s in range(1, self.max_number_lstm_layers + 1):
                parameters[f'LSTM_layer_{s}'] = hp.Int(f'LSTM_layer_{s}', min_value=32, max_value=256, step=16)
                
            for j in range(1, self.max_number_dense_layers + 1):
                parameters[f'dense_layer_{j}'] = hp.Int(f'dense_layer_{j}', min_value=32, max_value=256, step=16)
            

        elif (isinstance(parameters, dict)) and (all(key in parameters for key in param_keys)):
            
            self.parameters_included = True

        else:
            raise ValueError(f'"parameters" must be a dictionary containing all of the following keys: \n\n{param_keys}', flush = True)
        
        start_bias = tf.keras.initializers.Constant(self.initial_bias)
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.InputLayer(input_shape = self.input_shape)) #, batch_size = self.batch_size))
        model.add(tf.keras.layers.Masking(mask_value = self.mask_value))
                
        for s in range(1, parameters['number_of_lstm_layers'] + 1):

            if (s == 1):
                model.add(tf.keras.layers.LSTM(units=parameters[f'LSTM_layer_{s}'],
                                  dropout=parameters['dropout_rate'],
                                  recurrent_dropout=parameters['recurrent_dropout_rate'],
                                  bias_initializer=start_bias,
                                  return_sequences = s < parameters['number_of_lstm_layers']))

            else:
                model.add(tf.keras.layers.LSTM(units=parameters[f'LSTM_layer_{s}'],
                                  dropout=parameters['dropout_rate'],
                                  recurrent_dropout=parameters['recurrent_dropout_rate'],
                                  return_sequences = s < parameters['number_of_lstm_layers']))
                

        for j in range(1, parameters['number_of_dense_layers'] + 1):
            model.add(tf.keras.layers.Dense(units=parameters[f'dense_layer_{j}'],
                                            activation=parameters['dense_layers_activation_funcs'],))

        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        if self.optimizer.upper() == 'ADAM':
            # Notice momentum is not being used if the compiler is Adam
            opt=tf.keras.optimizers.Adam(learning_rate = parameters['learning_rate'])		

        elif self.optimizer.upper() == 'SGD':
            opt = tf.keras.optimizers.SGD(learning_rate = parameters['learning_rate'], 
                                          momentum = self.momentum)

        if self.metrics == None:
            model.compile(loss = self.convert_loss(), optimizer = opt)
        else:
            model.compile(loss = self.convert_loss(), optimizer = opt, metrics=self.metrics)

        return model
    
    def build(self, hp, parameters=None):
        self.parameters = parameters
        if self.strategy:
            with self.strategy.scope():
                if self.parameters:
                    model = self.build_model_body(hp=None, parameters = self.parameters)
                else:
                    model = self.build_model_body(hp)
        else:
            if self.parameters:
                model = self.build_model_body(hp=None, parameters = self.parameters)
            else:
                model = self.build_model_body(hp)
        return model

    def convert_to_TF_data_object(self, data, BATCH_SIZE):
        ''' Where data is a tuple with elements data_X and data_y
        OR just data_X'''
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.batch(BATCH_SIZE)
        data = data.with_options(options)
        return data
    
    def fit(self, hp, model, x, y, validation_data, **kwargs): # *args
        
        if self.parameters_included:
            batch_size = self.parameters['batch_size']*self.number_of_gpus
            batch_size = int(batch_size)
        else:
            batch_size = hp.get('batch_size')*self.number_of_gpus
            batch_size = int(batch_size)
                    
        train_data = self.convert_to_TF_data_object((x, y), batch_size) 
        validation_data = self.convert_to_TF_data_object(validation_data, batch_size) 
        
        # train_n_steps = len(train_data)
        # val_n_steps = len(validation_data)
        
        # train_n_steps = int(np.ceil(x.shape[0]/batch_size))
        # val_n_steps = int(np.ceil(validation_data[0].shape[0]/batch_size))
        
        return model.fit(
            train_data,
            validation_data = validation_data,
            # steps_per_epoch = train_n_steps,
            # validation_steps = val_n_steps,
            
            # Tune whether to shuffle the data in each epoch.
            # shuffle = hp.Boolean("shuffle"),
            batch_size = batch_size,
            # *args,
            **kwargs)
