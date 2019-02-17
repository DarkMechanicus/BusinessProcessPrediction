import os, datetime, time
import tensorflow as tf
import ensembles.accuracy_measures as acc

main_folder_name = 'models'

def prediction_correct(prediction_raw, i, args, dataset):
    _divisor = args['divisors'][6]
    _offset = args['offsets'][6]
    prediction = (prediction_raw[0] * _divisor) + _offset 
    ground_truth = args[dataset][6][i][0] + _offset
    ground_truth_plannedtimestamp = args[dataset][7][i][0] + _offset
    if ground_truth <= ground_truth_plannedtimestamp:
        return prediction <= ground_truth_plannedtimestamp
    else:
        return prediction > ground_truth_plannedtimestamp

def get_model_path(subfolder_name):
    now = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
    curr_path = main_folder_name
    if not os.path.exists(curr_path):
        os.mkdir(curr_path)

    curr_path += '/' + subfolder_name    
    if not os.path.exists(curr_path):
        os.mkdir(curr_path)

    curr_path += '/' + now
    if not os.path.exists(curr_path):
        os.mkdir(curr_path)
    
    return curr_path

def prepare_data(args):
    print('[Ensemble] Preparing test data...')
    test_x = []
    test_y = []
    for i in range(len(args['testdata'][0])):
        sequencelength = len(args['testdata'][0][i]) - 1 #minus eol character
        for prefix_size in range(1,sequencelength):   
            cropped_data = []
            for a in range(len(args['testdata'])):
                cropped_data.append(args['testdata'][a][i][:prefix_size])  
            prefix_activities = args['testdata'][0][i][:prefix_size]
            suffix_activities = args['testdata'][0][i][prefix_size:]
            if '!' in prefix_activities:
                        break # make no prediction for this case, since this case has ended already 

            ground_truth = args['testdata'][6][i][0] + args['offsets'][6] #undo offset
            ground_truth_plannedtimestamp = args['testdata'][7][i][0] + args['offsets'][7] #undo offset
            prepared_data = args['datadefinition'].EncodePrediction(cropped_data, args)
            prepared_truth = -1 if ground_truth <= ground_truth_plannedtimestamp else 1
            test_x.append(prepared_data)
            test_y.append({'binary': prepared_truth, 'ground_truth': ground_truth, 'planned': ground_truth_plannedtimestamp, 'id': i, "prefix": prefix_activities, "suffix": suffix_activities})

    return test_x, test_y

import numpy, csv
import glob
import utility.models as modelWrapper
class GenericEnsembleWrapper():
    def load_models(self, path, args):
        self.models = []
        for model_name in glob.glob('{}/*-model.h5'.format(path)):
            print('[Ensemble] Loading model {}...'.format(model_name))
            model = modelWrapper.CreateModel(args)
            model.load_weights('{}'.format(model_name))
            self.models.append(model)
        
        self.weights = numpy.full(len(self.models), 1.0)
        if os.path.exists('{}/ensemble_weights.csv'.format(path)):
            with open('{}/ensemble_weights.csv'.format(path), 'r', newline='') as weights_file:
                reader = csv.reader(weights_file, delimiter=',', quotechar='|')
                next(reader, None)
                for row in reader:
                    self.weights[int(row[0])] = float(row[1])
        self.path = path

    def evaluate(self, samples, solutions, args, ensemble_type, ensemble, pruningParams, original_size, pruned_size):
        confusion_matrix = {
            'tp': 0,
            'fp': 0,
            'tn': 0,
            'fn': 0,
            'total': 0
        }

        # open save file, so we can write to the file when we get the results
        with open('../ensemble_results.csv'.format(), 'a', newline='') as result_file:
            writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            loop_start = time.time()
            print("Getting {} predictions...".format(len(samples)))
            for i in range(len(samples)):
                test_data = samples[i]
                test_solution = solutions[i]
                ensemble_prediction = self.__predict(test_data, args)
                ensemble_binary = -1 if ensemble_prediction < test_solution['planned'] else 1
                result = {
                    'type': ensemble_type,
                    'ensemble': ensemble,
                    'id': test_solution['id'],
                    'a_m': pruningParams['a_m'],
                    'a_l_b': pruningParams['a_l_b'],
                    'd_m': pruningParams['d_m'],
                    'd_l_b': pruningParams['d_l_b'],
                    'original_size': original_size,
                    'pruned_size': pruned_size,
                    'e_p': ensemble_prediction,
                    'e_p_b': ensemble_binary,
                    'g_t': test_solution['ground_truth'],
                    'g_t_pts': test_solution['planned'],
                    'g_t_b': test_solution['binary'],
                    'prefix': ' '.join(test_solution['prefix']),
                    'suffix': ' '.join(test_solution['suffix'])
                }
                writer.writerow(result.values())
                if result['e_p_b'] == result['g_t_b']:
                    confusion_matrix['tp'] += (1 if result['e_p_b'] > 0 else 0)
                    confusion_matrix['tn'] += (1 if result['e_p_b'] < 0 else 0)
                else:
                    confusion_matrix['fp'] += (1 if result['e_p_b'] > 0 else 0)
                    confusion_matrix['fn'] += (1 if result['e_p_b'] < 0 else 0)
                confusion_matrix['total'] += 1
            
            print(time.time() - loop_start)

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

        #save accuracy measures
        with open('../ensemble_accuracy_measurements.csv'.format(), 'a', newline='') as acc_file:
            writer = csv.writer(acc_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow([ensemble_type, ensemble, pruningParams['a_m'], pruningParams['a_l_b'], pruningParams['d_m'], pruningParams['d_l_b']] + list(acc_results.values()))

    def __predict(self, data, args):
        graph = tf.get_default_graph()
        with graph.as_default():
            weighted_predictions = []
            for i in range(len(self.models)):
                model = self.models[i]
                weight = self.weights[i]
                prediction = model.predict(data, verbose=0)[0][0]
                prediction = (prediction * args['divisors'][6]) + args['offsets'][6]
                weighted_predictions.append(prediction * weight)
        
        ensemble_prediction = sum(weighted_predictions) / len(weighted_predictions)

        return ensemble_prediction
