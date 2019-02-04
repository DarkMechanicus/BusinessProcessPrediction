import keras as keras_impl
import numpy as np
import copy
import tensorflow as tf
import os
import utility.models as models
import random, math, datetime, csv, glob

class AdaBoostWrapper(keras_impl.callbacks.Callback):
    resampling = False
    weights = []
    train_matrices = []
    validation_matrices = []
    ensemble_weights = []
    ensemble_models = []
    args = None

    def __init__(self, resampling = False):
        self.resampling = resampling

    def init_weights(self):
        if len(self.weights) > 0:
            return
        
        lenght = len(self.train_matrices['X'])
        for i in range(lenght):
            self.weights.append(1/lenght)

        self.weights = np.asarray(self.weights)

        print('[AdaBoost] Init {} weights'.format(len(self.weights)))
        return 

    def prediction_correct(self, prediction, i, dataset = 'train_sentences'):
        ground_truth = self.args[dataset][6][i][0] + self.args['offsets'][6]
        ground_truth_plannedtimestamp = self.args[dataset][7][i][0] + self.args['offsets'][7]

        if ground_truth <= ground_truth_plannedtimestamp:
            return prediction <= ground_truth_plannedtimestamp
        else:
            return prediction > ground_truth_plannedtimestamp

    def do_resample(self):
        resampling_size = 100
        total_weight = math.ceil(sum(self.weights) * resampling_size / len(self.train_matrices['X']))
        weighted_sample = []
        for i in range(len(self.train_matrices['X'])):
            for j in range(math.ceil(self.weights[i]* resampling_size)):
                weighted_sample.append({'X': self.train_matrices['X'][i], 'y_t': self.train_matrices['y_t'][i]})

        random.shuffle(weighted_sample)

        resampled = weighted_sample[::total_weight]
        print('Took {} elements, scaled to {}, resampled to {}'.format(len(self.train_matrices['X']), len(weighted_sample), len(resampled)))
        train_x = []
        train_y = []
        for sample in resampled:
            train_x.append(sample['X'])
            train_y.append(sample['y_t'])
        
        return {'X': train_x, 'y_t': train_y}
           
    def post_training(self):
        print('[AdaBoost] Predictions for validation data...')
        predictions = self.model.predict(self.train_matrices['X'])
        simple_error = []
        weighted_error = []
        print('[AdaBoost] Calculate error for validation data...')
        for i in range(len(predictions)):
            prediction = (predictions[i] * self.args['divisors'][6]) + self.args['offsets'][6]
            correct = self.prediction_correct(prediction, i)
            simple_error.append(0 if correct == True else 1)
            weighted_error.append((0 if correct == True else 1) * self.weights[i])
        
        total_error = sum(weighted_error) / sum(self.weights)
        temp = (1 - total_error) / total_error;
        if temp <= 0.0:
            temp = 0.000000001 

        stage = np.log([temp])[0]

        print('[AdaBoost] This model has a weight of {} and a total error of {}'.format(stage, total_error))        
        self.ensemble_weights.append(stage);

        print('[AdaBoost] Modifing weights...')
        for i in range(len(self.weights)):
            new_weight = self.weights[i] * np.exp(stage * simple_error[i]);
            self.weights[i] = new_weight

        print('[AdaBoost] Post training modifications done!')
        return

    def on_train_end(self, epoch, logs={}):
        self.post_training()
        return

    def predict(self, data):
        graph = tf.get_default_graph()
        with graph.as_default():
            weighted_predictions = []
            for i in range(len(self.ensemble_models)):
                model = self.ensemble_models[i]
                prediction = model.predict(data, verbose=0)
                weighted_predictions.append(prediction * self.ensemble_weights[i])
        
        ensemble_prediction = sum(weighted_predictions)

        return ensemble_prediction

    def save_weights(self, datestr):
        with open('models/{}/ensemble_weights.csv'.format(datestr), 'w', newline='') as weights_file:
            writer = csv.writer(weights_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["model_id", "weight"])
            for i in range(len(self.ensemble_weights)):
                model_id = i
                weight = self.ensemble_weights[i]
                writer.writerow([model_id, weight])
    
    def load_ensemble(self, datestr):
        self.ensemble_models = []
        for model_name in glob.glob('models/{}/*-model.h5'.format(datestr)):
            print('[AdaBoost] Loading model {}...'.format(model_name))
            model = models.CreateModel(self.args)
            model.load_weights('{}'.format(model_name))
            self.ensemble_models.append(model)
        
        self.ensemble_weights = np.full(len(self.ensemble_models), 1)
        with open('models/{}/ensemble_weights.csv'.format(datestr), 'r', newline='') as weights_file:
            reader = csv.reader(weights_file, delimiter=',', quotechar='|')
            next(reader, None)
            for row in reader:
                self.ensemble_weights[int(row[0])] = float(row[1])

    def save_result(self, datestr, results):
        with open('models/{}/ensemble_result.csv'.format(datestr), 'w', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["prediction", "ground_truth"])
            for result in results:
                writer.writerow([result['prediction'], result['ground_truth']])
