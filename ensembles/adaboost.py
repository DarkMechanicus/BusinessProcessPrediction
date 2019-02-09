import keras as keras_impl
import numpy as np
import copy
import tensorflow as tf
import os
import utility.models as models
import random, math, datetime, csv, glob
import ensembles.accuracy_measures as acc
from ensembles.ensemble import prediction_correct

class AdaBoostWrapper(keras_impl.callbacks.Callback):
    weights = []
    train_matrices = {}
    validation_matrices = {}
    ensemble_weights = []
    args = {}
    sequence_weight = []
    val_weights = []

    def __init__(self, resampling, can_weights_decrease, modify_validation_data):
        self.resampling = resampling
        self.can_weights_decrease = can_weights_decrease
        self.modify_validation_data = modify_validation_data

    def init_weights(self):
        if len(self.weights) > 0:
            return
        
        length = len(self.train_matrices['X'])
        seq_length = len(self.args['traindata'][0])
        val_length = len(self.validation_matrices['X'])

        self.weights = np.full(length, 1/length)
        self.sequence_weight = np.full(seq_length, 1/seq_length)
        self.val_weights = np.full(val_length, 1/val_length)

        print('[AdaBoost] Init {} weights'.format(len(self.weights)))
        return 

    def do_resample(self, data):
        print('[AdaBoost] Resampling...')
        # based on regularization.ShuffleArray
        state = np.random.get_state()
        sample_data = []
        print('Sequence Weights: ', self.sequence_weight)
        # resample in a away that gives the smallest weight exactly 1 sample
        avg_weight = sum(self.sequence_weight) / len(self.sequence_weight)
        print('Avg weight', avg_weight)
        resample_size = math.ceil(10 / avg_weight)
        total_weight = math.ceil(sum(self.sequence_weight) * resample_size / len(data))
        print('Resample Size: ', resample_size)
        print('Total Weight: ', total_weight)
        print('Datesize Pre: ', len(data))
        for i in range(len(data)):
            scaled_data = []
            for j in range(len(data[i])):
                scaled_count = math.ceil(self.sequence_weight[j] * resample_size)
                for k in range(scaled_count):
                    scaled_data.append(data[i])
                np.random.shuffle(scaled_data)
                scaled_data = scaled_data[::total_weight]
                np.random.set_state(state)
            sample_data.append(scaled_data[0])
            print('Scaled Size: ', len(scaled_data))
        
        print('Datasize Post: ', len(sample_data))
        return sample_data
           
    def __adjust_training_weights(self):
        print('[AdaBoost] Predictions for training data...')
        predictions = self.model.predict(self.train_matrices['X'])
        simple_error = []
        weighted_error = []
        print('[AdaBoost] Calcualte error for training sequences...')
        sample = 0
        for i in range(len(self.args['traindata'][0])):
            seq_len = len(self.args['traindata'][0][i])
            seq_simple_errors = []
            seq_weighted_errors = []

            for j in range(1, seq_len):
                prediction = predictions[sample] 
                correct = prediction_correct(prediction, sample, self.args, 'train_sentences')
                correct = (0 if correct == True else 1)
                seq_simple_errors.append(correct)
                seq_weighted_errors.append(correct * self.weights[sample])
                sample += 1

            # average errors over the entire sequence
            seq_correct = sum(seq_simple_errors) / len(seq_simple_errors)
            seq_correct = 0 if (seq_correct < 0.5) else 1
            seq_weight = sum(seq_weighted_errors) / len(seq_weighted_errors)
            self.sequence_weight[i] = seq_weight

            for j in range(1, seq_len):
                simple_error.append(seq_correct)
                weighted_error.append(seq_correct * seq_weight)

        total_error = sum(weighted_error) / sum(self.weights)
        temp = (1 - total_error) / total_error
        if temp <= 0.0:
            temp = 0.000000001 

        stage = 0.5 * np.log([temp])[0]

        print('[AdaBoost] This model has a weight of {} and a total error of {}'.format(stage, total_error))        
        self.ensemble_weights.append(stage)

        print('[AdaBoost] Modifing training weights...')
        for i in range(len(self.weights)):  
            modificator = simple_error[i]
            if self.can_weights_decrease:
                modificator = -1 if modificator == 0 else modificator         
            new_weight = self.weights[i] * np.exp((-stage) * modificator)
            self.weights[i] = new_weight

        return

    def __adjust_validation_weights(self):
        print('[AdaBoost] Predictions for validation data...')
        predictions = self.model.predict(self.validation_matrices['X'])
        simple_error = []
        weighted_error = []
        print('[AdaBoost] Calcualte error for validation sequences...')
        sample = 0
        _divisor = self.args['divisors'][6]
        _offset = self.args['offsets'][6]
        for i in range(len(self.args['validationdata'][0])):
            seq_len = len(self.args['validationdata'][0][i])
            seq_simple_errors = []
            seq_weighted_errors = []

            for j in range(1, seq_len):
                prediction = (predictions[sample] * _divisor) + _offset
                correct = prediction_correct(prediction, sample, self.args, 'validation_sentences')
                correct = (0 if correct == True else 1)
                seq_simple_errors.append(correct)
                seq_weighted_errors.append(correct * self.val_weights[sample])
                sample += 1

            # average errors over the entire sequence
            seq_correct = sum(seq_simple_errors) / len(seq_simple_errors)
            seq_correct = 0 if (seq_correct < 0.5) else 1
            seq_weight = sum(seq_weighted_errors) / len(seq_weighted_errors)

            for j in range(1, seq_len):
                simple_error.append(seq_correct)
                weighted_error.append(seq_correct * seq_weight)

        total_error = sum(weighted_error) / sum(self.val_weights)
        temp = (1 - total_error) / total_error
        if temp <= 0.0:
            temp = 0.000000001 

        stage = 0.5 * np.log([temp])[0]

        print('[AdaBoost] Modifing validation weights...')
        for i in range(len(self.val_weights)):  
            modificator = simple_error[i]
            if self.can_weights_decrease:
                modificator = -1 if modificator == 0 else modificator         
            new_weight = self.val_weights[i] * np.exp((-stage) * modificator)
            self.val_weights[i] = new_weight

        return

    def on_train_end(self, epoch, logs={}):
        self.__adjust_training_weights()
        if self.modify_validation_data == True:
            self.__adjust_validation_weights()
        print('[AdaBoost] Post training modifications done!')
        return

    def on_train_begin(self, epoch, logs={}):
        self.__save_model_weights()
        return

    def __save_model_weights(self):
        with open('{}-sample_weights.csv'.format(self.args['running']), 'w', newline='') as weights_file:
            writer = csv.writer(weights_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['sample_nr', 'sample_weight', 'sequence_id', 'prefix', 'suffix'])
            sample = 0
            for i in range(len(self.args['testdata'][0])):
                sequencelength = len(self.args['testdata'][0][i]) - 1 #minus eol character
                ground_truth_processid = self.args['testdata'][8][i][0]
                for prefix_size in range(1,sequencelength):   
                    prefix_activities = ' '.join(self.args['testdata'][0][i][:prefix_size])
                    suffix_activities = ' '.join(self.args['testdata'][0][i][prefix_size:])
                    sample += 1
                writer.writerow([sample, self.weights[sample], ground_truth_processid, prefix_activities, suffix_activities])                

    def save_ensemble_weights(self, path):
        with open('{}/ensemble_weights.csv'.format(path), 'w', newline='') as weights_file:
            writer = csv.writer(weights_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["model_id", "weight"])
            for i in range(len(self.ensemble_weights)):
                model_id = i
                weight = self.ensemble_weights[i]
                writer.writerow([model_id, weight])
     
