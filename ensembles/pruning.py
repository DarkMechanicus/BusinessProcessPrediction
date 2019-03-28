from enum import Enum
class DiversityPruningMethods(Enum):
    QStatistics = "q_statistics"
    CorrelationCoefficient = "correlation_coefficient"
    DisagreementMeasure = "disagreement_measure"
    DoubleFaultMeasure = "double_fault_measure"

class AccuracyPruningMethods(Enum):
    MCC = "mcc"
    Precision = "precision"
    Recall = "recall"
    Specificity = "specificity"
    FalsePositiveRate = "false_positve_rate"
    NegativePredictionValue = "negative_prediction_value"
    Accuracy = "accuracy"
    F1 = "f1"

import numpy as np
import ensembles.accuracy_measures as acc
import csv, os
from ensembles.ensemble import prediction_correct

class PruningWrapper:
    def __init__(self, pruningParams, args):
        self.accuracy_lower_bound = pruningParams['a_l_b']
        self.accuracy_method = AccuracyPruningMethods(pruningParams['a_m'])
        self.diversity_lower_bound = pruningParams['d_l_b']
        self.diversity_method = DiversityPruningMethods(pruningParams['d_m'])
        self.args = args

    def do_diversity_measure_computation(self, models, model_weights, data, solutions, path, predictions):
        for i in range(len(models)):
            if(os.path.isfile('{}/{:03}-model_diversity_measures.csv'.format(models, i))):
                continue

            with open('{}/{:03}-model_diversity_measures.csv'.format(path, i), 'a', newline='') as result_file:
                writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                measures = []
                for j in range(len(models)):
                    if i == j:
                        continue
                    matrix = self.__compute_corecctness_matrix(i, j, data, solutions, predictions)
                    measure = self.__compute_diversity_measure(matrix)
                    measures.append(measure)
                    writer.writerow(["{:03}-model".format(j), self.diversity_method.value, measure] + list(matrix.values()))

    def do_accuracy_measure_computation(self, models, model_weights, data, solutions, path, predictions):
        for i in range(len(models)):
            if(os.path.isfile('{}/{:03}-model_accuracy_measures.csv'.format(models, i))):
                continue

            with open('{}/{:03}-model_accuracy_measures.csv'.format(path, i), 'a', newline='') as result_file:
                writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                matrix = self.__compute_prediction_matrix(i, data, solutions, predictions)
                metric = self.__compute_accuracy_measure(matrix)
                writer.writerow([self.accuracy_method.value, metric] + list(matrix.values()))

    def do_pruning(self, models, model_weights, model_ids, data, solutions, path):
        accuracy_pruned = self.__accuracy_pruning(models, model_weights, model_ids, data, solutions, path)
        diversity_pruned = self.__diversity_pruning(accuracy_pruned['models'], accuracy_pruned['weights'], accuracy_pruned['ids'], data, solutions, path)
        
        return diversity_pruned

    def __compute_corecctness_matrix(self, model_one, model_two, data, solutions, predictions):
        matrix = {
            'both_correct': 0,
            'one_correct': 0,
            'two_correct': 0,
            'neither_correct': 0
        }

        predictions_one = predictions[model_one]
        predictions_two = predictions[model_two]

        for i in range(len(predictions_one)):
            p_one = prediction_correct(predictions_one[i][0], i, self.args, 'test_sentences')
            p_two = prediction_correct(predictions_two[i][0], i, self.args, 'test_sentences')
            sol = True if solutions[i]['binary'] == 0 else False
            if p_one == p_two:
                matrix['both_correct'] = matrix['both_correct'] + (1 if p_one == sol else 0)
                matrix['neither_correct'] = matrix['neither_correct'] + (0 if p_one == sol else 1)
            else:
                matrix['one_correct'] = matrix['one_correct'] + (1 if p_one == sol else 0)
                matrix['two_correct'] = matrix['two_correct'] + (1 if p_two == sol else 0)
            
        matrix['total'] = matrix['both_correct'] + matrix['neither_correct'] + matrix['one_correct'] + matrix['two_correct']

        return matrix
    
    def __diversity_pruning(self, models, model_weights, model_ids, data, solutions, path):
        if self.diversity_lower_bound <= 0.0:
            return {'models': models, 'weights': model_weights, 'ids': model_ids}          

        pruned_models = []
        pruned_model_ids = []
        pruned_weights = []
        for i in range(len(models)):
            with open('{}/{:03}-model_diversity_measures.csv'.format(path, i), 'r', newline='') as result_file:
                reader = csv.reader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                data_rows = list(filter(lambda line: line[1] == self.diversity_method.value, reader))
                measures = []
                for row in data_rows:
                    if not int(row[0][0:3]) in model_ids:
                        print(row[0], 'skipped')
                        break
                    measures.append(float(row[2]))

                if len(measures) == 0:
                    print("No measurements found!")
                    continue

            avg_diversity = sum(measures) / len(measures)
            print('Model {} has avg. diversity of {}. Will {}be removed!'.format(i, avg_diversity, 'not ' if avg_diversity > self.diversity_lower_bound else ''))
            if avg_diversity > self.diversity_lower_bound:
                pruned_models.append(models[i])
                pruned_weights.append(model_weights[i])
                pruned_model_ids.append(model_ids[i])
        
        return {'models': pruned_models, 'weights': pruned_weights, 'ids': pruned_model_ids}        

    def __compute_diversity_measure(self, matrix):
        if self.diversity_method == DiversityPruningMethods.DisagreementMeasure:
                return (matrix['one_correct'] + matrix['two_correct']) / matrix['total']
        elif self.diversity_method == DiversityPruningMethods.DoubleFaultMeasure:
                result = (matrix['neither_correct'] / matrix['total'])
                normResult = (-result) + 1
                return normResult
        elif self.diversity_method == DiversityPruningMethods.CorrelationCoefficient:
                top = (matrix['both_correct'] * matrix['neither_correct']) - (matrix['one_correct'] * matrix['two_correct'])
                bottom_total = (matrix['both_correct'] + matrix['one_correct']) * (matrix['two_correct'] + matrix['neither_correct']) * (matrix['both_correct'] + matrix['two_correct']) * (matrix['one_correct'] + matrix['neither_correct'])
                bottom = np.sqrt(bottom_total)
                result = (top / bottom)
                normResult = (result * -0.5) + 0.5
                return normResult
        elif self.diversity_method == DiversityPruningMethods.QStatistics:
                a = (matrix['both_correct'] * matrix['neither_correct'])
                b = (matrix['one_correct'] * matrix['two_correct'])
                result = ((a - b) / (a + b))
                normResult = (result * -0.5) + 0.5
                return normResult
        else:
                raise ValueError("unknown value for diviersity method: {}".format(self.diversity_method))

    def __compute_accuracy_measure(self, matrix):
        if self.accuracy_method == AccuracyPruningMethods.Precision:
            return acc.compute_precision(matrix)
        elif self.accuracy_method == AccuracyPruningMethods.Recall:
            return acc.compute_recall(matrix)
        elif self.accuracy_method == AccuracyPruningMethods.Specificity:
            return acc.compute_specificity(matrix)
        elif self.accuracy_method == AccuracyPruningMethods.FalsePositiveRate:
            return acc.compute_false_positive_rate(matrix)
        elif self.accuracy_method == AccuracyPruningMethods.NegativePredictionValue:
            return acc.compute_negative_prediction_value(matrix)
        elif self.accuracy_method == AccuracyPruningMethods.Accuracy:
            return acc.compute_accuracy(matrix)
        elif self.accuracy_method == AccuracyPruningMethods.F1:
            return acc.compute_f1(matrix)
        elif self.accuracy_method == AccuracyPruningMethods.MCC:
            return acc.compute_mcc(matrix)
        else:
            raise ValueError("unknown value for accuracy method: {}".format(self.accuracy_method))

    def __accuracy_pruning(self, models, model_weights, model_ids, data, solutions, path):
        if self.accuracy_lower_bound <= 0.0:
            return {'models': models, 'weights': model_weights, 'ids': model_ids}
        
        pruned_modles=[]
        pruned_model_ids = []
        pruned_weights=[]
        for i in range(len(models)):
            metric = np.nan
            with open('{}/{:03}-model_accuracy_measures.csv'.format(path, i), 'r', newline='') as result_file:
                reader = csv.reader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                data_row = list(filter(lambda line: line[0] == self.accuracy_method.value, reader))[0]
                metric = float(data_row[1])

            if metric == np.nan or metric == np.inf:
                raise ValueError("{} as metric {}".format(metric, self.accuracy_method))
            print('Model {} has {} of {}. Will {}be removed!'.format(i, self.accuracy_method, metric, 'not ' if metric > self.accuracy_lower_bound else ''))
            if metric > self.accuracy_lower_bound:
                pruned_modles.append(models[i])
                pruned_model_ids.append(model_ids[i])
                pruned_weights.append(model_weights[i])
        
        return {'models': pruned_modles, 'weights': pruned_weights, 'ids': pruned_model_ids}

    def __compute_prediction_matrix(self, model, data, solutions, predictions):
        matrix = {
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }
        
        predictions_one = predictions[model]

        for i in range(len(predictions_one)):
            prediction =  prediction_correct(predictions_one[i], i, self.args, 'test_sentences')[0]
            solution = True if solutions[i]['binary'] == 1 else False

            if (prediction == True and solution == True):
                matrix['tp'] = matrix['tp'] + 1
            elif (prediction == True and solution == False):
                matrix['fp'] = matrix['fp'] + 1
            elif (prediction == False and solution == False):
                matrix['tn'] = matrix['tn'] + 1
            elif (prediction == False and solution == True):
                matrix['fn'] = matrix['fn'] + 1
            else:
                raise Exception("unreachable code!")
        
        matrix['total'] = matrix['tp'] + matrix['tn'] + matrix['fn'] + matrix['fp']

        return matrix

            