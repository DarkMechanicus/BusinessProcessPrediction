import keras as keras_impl
import numpy as np
import copy
import tensorflow as tf
import os
import utility.models as models
import random, math, datetime, csv, glob
import utility.accuracy_measures as acc

class AdaBoostWrapper(keras_impl.callbacks.Callback):
    resampling = False
    weights = []
    train_matrices = []
    ensemble_weights = []
    ensemble_models = []
    args = None
    sequence_weight = []

    def __init__(self, resampling = False):
        self.resampling = resampling

    def init_weights(self):
        if len(self.weights) > 0:
            return
        
        length = len(self.train_matrices['X'])
        seq_length = len(self.args['traindata'][0])
        #for i in range(lenght):
        #    self.weights.append(1/lenght)
        #self.weights = np.asarray(self.weights)

        self.weights = np.full(length, 1/length)
        self.sequence_weight = np.full(seq_length, 1/seq_length)

        print('[AdaBoost] Init {} weights'.format(len(self.weights)))
        return 

    def prediction_correct(self, prediction, i, dataset = 'train_sentences'):
        ground_truth = self.args[dataset][6][i][0] + self.args['offsets'][6]
        ground_truth_plannedtimestamp = self.args[dataset][7][i][0] + self.args['offsets'][7]
        #print('Prediction {} GT {} Planned {}'.format(prediction[0], ground_truth, ground_truth_plannedtimestamp))
        if ground_truth <= ground_truth_plannedtimestamp:
            return prediction[0] <= ground_truth_plannedtimestamp
        else:
            return prediction[0] > ground_truth_plannedtimestamp

    def do_resample(self):
        print('[AdaBoost] Resampling...')
        # based on regularization.ShuffleArray
        resampling_size = 100
        data = self.args['traindata']
        state = np.random.get_state()
        sample_data = []
        total_weight = math.ceil(sum(self.sequence_weight) * resampling_size / len(data))
        for i in range(len(data)):
            scaled_data = []
            for j in range(len(data[i])):
                scaled_count = math.ceil(self.sequence_weight[j] * resampling_size)
                for k in range(scaled_count):
                    scaled_data.append(data[i])
                np.random.shuffle(scaled_data)
                scaled_data = scaled_data[::total_weight]
                np.random.set_state(state)
            sample_data.append(scaled_data[0])
        return sample_data
           
    def post_training(self):
        print('[AdaBoost] Predictions for training data...')
        predictions = self.model.predict(self.train_matrices['X'])
        simple_error = []
        weighted_error = []
        print('[AdaBoost] Calcualte error for training sequences...')
        sample = 0
        _divisor = self.args['divisors'][6]
        _offset = self.args['offsets'][6]
        for i in range(len(self.args['traindata'][0])):
            seq_len = len(self.args['traindata'][0][i])
            seq_simple_errors = []
            seq_weighted_errors = []

            for j in range(1, seq_len):
                prediction = (predictions[sample] * _divisor) + _offset
                correct = self.prediction_correct(prediction, sample)
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
        temp = (1 - total_error) / total_error;
        if temp <= 0.0:
            temp = 0.000000001 

        stage = 0.5 * np.log([temp])[0]

        print('[AdaBoost] This model has a weight of {} and a total error of {}'.format(stage, total_error))        
        self.ensemble_weights.append(stage);

        print('[AdaBoost] Modifing weights...')
        for i in range(len(self.weights)):
            new_weight = self.weights[i] * np.exp((-stage) * simple_error[i]);
            self.weights[i] = new_weight

        print('[AdaBoost] Post training modifications done!')
        return

    def on_train_end(self, epoch, logs={}):
        self.post_training()
        return

    def evalute(self, test_x, test_y, datestr):
        results = []
        confusion_matrix = {
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
            'total': 0
        }
        for i in range(len(test_x)):
            test_data = test_x[i]
            test_solution = test_y[i]
            ensemble_prediction = self.__predict(test_data)
            ensemble_binary = -1 if ensemble_prediction < test_solution['planned'] else 1
            #print('EP: {}\nGT: {}\nPT: {}\nEB: {}\nBi: {}'.format(ensemble_prediction, test_solution['ground_truth'], test_solution['planned'], ensemble_binary, test_solution['binary']))
            result = {
                'id': test_solution['id'],
                'e_p': ensemble_prediction,
                'e_p_b': ensemble_binary,
                'g_t': test_solution['ground_truth'],
                'g_t_pts': test_solution['planned'],
                'g_t_b': test_solution['binary'],
                'prefix': ' '.join(test_solution['prefix']),
                'suffix': ' '.join(test_solution['suffix'])
            }
            results.append(result)
            if result['e_p_b'] == result['g_t_b']:
                confusion_matrix['tp'] += (1 if result['e_p_b'] > 0 else 0)
                confusion_matrix['tn'] += (1 if result['e_p_b'] < 0 else 0)
            else:
                confusion_matrix['fp'] += (1 if result['e_p_b'] > 0 else 0)
                confusion_matrix['fn'] += (1 if result['e_p_b'] < 0 else 0)
            confusion_matrix['total'] += 1

        # compute accuracy measures
        acc_results = {
            'precision': acc.compute_precision(confusion_matrix),
            'recall': acc.compute_recall(confusion_matrix),
            'specificity': acc.compute_specificity(confusion_matrix),
            'false_positive_rate': acc.compute_false_positive_rate(confusion_matrix),
            'negative_prediction_value': acc.compute_negative_prediction_value(confusion_matrix),
            'accuracy': acc.compute_accuracy(confusion_matrix),
            'f1': acc.compute_f1(confusion_matrix),
            'mcc': acc.compute_mcc(confusion_matrix)
        }

        print(acc_results)

        #save ensemble results
        self.__save_result(datestr, results)
        self.__save_accuracy(datestr, acc_results)

    def __predict(self, data):
        graph = tf.get_default_graph()
        with graph.as_default():
            weighted_predictions = []
            for i in range(len(self.ensemble_models)):
                model = self.ensemble_models[i]
                weight = self.ensemble_weights[i]
                prediction = model.predict(data, verbose=0)[0][0]
                prediction = (prediction * self.args['divisors'][6]) + self.args['offsets'][6]
                weighted_predictions.append(prediction * weight)
        
        ensemble_prediction = sum(weighted_predictions) / len(weighted_predictions)

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
        
        self.ensemble_weights = np.full(len(self.ensemble_models), 1.0)
        with open('models/{}/ensemble_weights.csv'.format(datestr), 'r', newline='') as weights_file:
            reader = csv.reader(weights_file, delimiter=',', quotechar='|')
            next(reader, None)
            for row in reader:
                self.ensemble_weights[int(row[0])] = float(row[1])

    def __save_result(self, datestr, results):
        with open('models/{}/ensemble_result.csv'.format(datestr), 'w', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["sequence_id", "ensemble_prediction", "ensemble_prediction_binary", "ground_truth", "ground_truth_plannedtimestamp", "ground_truth_binary", "prefix", "suffix"])
            for result in results:
                writer.writerow(result.values())

    def __save_accuracy(self, datestr, results):
        with open('models/{}/ensemble_accuracy.csv'.format(datestr), 'w', newline='') as acc_file:
            writer = csv.writer(acc_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['precision', 'recall', 'specificity', 'false_positive_rate', 'negative_prediction_value', 'accuracy', 'f1', 'mcc'])
            writer.writerow(results.values())        
