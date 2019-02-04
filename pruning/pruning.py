from enum import Enum
class DiversityPruningMethods(Enum):
    QStatistics = "q_statistics"
    CorrelationCoefficient = "correlation_coefficient"
    DisagreementMeasure = "disagreement_measure"
    DoubleFaultMeasure = "double_fault_measure"

import numpy as np

class PruningWrapper:

    accuracy_lower_bound = 0.0
    diversity_lower_bound = 0.0
    method = DiversityPruningMethods.DisagreementMeasure

    def __init__(self, accuracy_lower_bound = 0.0, diversity_lower_bound = 0.0, method = DiversityPruningMethods.DisagreementMeasure):
        self.accuracy_lower_bound = accuracy_lower_bound
        self.diversity_lower_bound = diversity_lower_bound
        self.method = method

    def do_pruning(self, models, model_weights, data, solutions):
        diversity_pruned = self.__diversity_pruning(models, model_weights, data, solutions)
        accuracy_pruned = self.__accuracy_pruning(diversity_pruned['models'], diversity_pruned['weights'], data, solutions)
        return accuracy_pruned

    def __compute_corecctness_matrix(self, model_one, model_two, data, solutions):
        matrix = {
            'both_correct': 0,
            'one_correct': 0,
            'two_correct': 0,
            'neither_correct': 0
        }

        predictions_one = []
        predictions_two = []
        for x in data:
            predictions_one.append(model_one.predict(x))
            predictions_two.append(model_two.predict(x))

        for i in range(len(predictions_one)):
            p_one = predictions_one[i][0]
            p_two = predictions_two[i][0]
            p_one = -1 if p_one < 0 else 1
            p_two = -1 if p_two < 0 else 1
            sol = solutions[i]
            if p_one == p_two:
                matrix['both_correct'] = matrix['both_correct'] + (1 if p_one == sol else 0)
                matrix['neither_correct'] = matrix['neither_correct'] + (0 if p_one == sol else 1)
            else:
                matrix['one_correct'] = matrix['one_correct'] + (1 if p_one == sol else 0)
                matrix['two_correct'] = matrix['two_correct'] + (1 if p_two == sol else 0)
            
        matrix['total'] = matrix['both_correct'] + matrix['neither_correct'] + matrix['one_correct'] + matrix['two_correct']

        return matrix
    
    def __diversity_pruning(self, models, model_weights, data, solutions):
        if self.diversity_lower_bound <= 0.0:
            return {'models': models, 'weights': model_weights}
        
        pruned_models = []
        pruned_weights = []
        for i in range(len(models) - 1):
            measures = []
            for j in range(1, len(models)):
                matrix = self.__compute_corecctness_matrix(models[i], models[j], data, solutions)
                measure = self.__compute_diversity_measure(matrix)
                measures.append(measure)
            
            avg_diversity = sum(measures) / len(measures)
            print('Model {} has avg. diversity of {}. Will {}be removed!'.format(i, avg_diversity, 'not ' if avg_diversity > self.diversity_lower_bound else ''))
            if avg_diversity > self.diversity_lower_bound:
                pruned_models.append(models[i])
                pruned_weights.append(model_weights[i])
        
        return {'models': pruned_models, 'weights': pruned_weights}        

    def __compute_diversity_measure(self, matrix):
        if self.method == DiversityPruningMethods.DisagreementMeasure:
                return (matrix['one_correct'] + matrix['two_correct']) / matrix['total']
        elif self.method == DiversityPruningMethods.DoubleFaultMeasure:
                return matrix['neither_correct'] / matrix['total']
        elif self.method == DiversityPruningMethods.CorrelationCoefficient:
                top = (matrix['both_correct'] * matrix['neither_correct']) - (matrix['one_correct'] * matrix['two_correct'])
                bottom_total = (matrix['both_correct'] + matrix['one_correct']) * (matrix['two_correct'] + matrix['neither_correct']) * (matrix['both_correct'] + matrix['two_correct']) * (matrix['one_correct'] + matrix['neither_correct'])
                bottom = np.sqrt(bottom_total)
                return top / bottom
        elif self.method == DiversityPruningMethods.QStatistics:
                a = (matrix['both_correct'] * matrix['neither_correct'])
                b = (matrix['one_correct'] * matrix['two_correct'])
                return (a - b) / (a + b)
        else:
                raise ValueError("unknown value for diviersity method: {}".format(self.method))
                    
    def __accuracy_pruning(self, models, model_weights, data, solutions):
        if self.accuracy_lower_bound <= 0.0:
            return {'models': models, 'weights': model_weights}
        
        pruned_modles=[]
        pruned_weights=[]
        for i in range(len(models)):
            model = models[i]
            matrix = self.__compute_prediction_matrix(model, data, solutions)
            mcc_top = (matrix['true_positive'] * matrix['true_negative']) - (matrix['false_positive'] * matrix['false_negative'])
            mcc_bottom_total = (matrix['true_positive'] + matrix['false_negative']) * (matrix['true_positive'] + matrix['false_positive']) * (matrix['true_negative'] + matrix['false_positive']) * (matrix['true_negative'] + matrix['false_negative'])
            mcc_bottom = np.sqrt(mcc_bottom_total)
            mcc = 0
            if mcc_top != 0:
                mcc = mcc_top / mcc_bottom
            print('Model {} has mcc of {}. Will {}be removed!'.format(i, mcc, 'not ' if mcc > self.accuracy_lower_bound else ''))
            if mcc > self.accuracy_lower_bound:
                pruned_modles.append(model)
                pruned_weights.append(model_weights[i])
        
        return {'models': pruned_modles, 'weights': pruned_weights}

    def __compute_prediction_matrix(self, model, data, solutions):
        matrix = {
            'true_positive': 0,
            'true_negative': 0,
            'false_positive': 0,
            'false_negative': 0
        }
        
        predictions = []
        for x in data:
         predictions.append(model.predict(x))

        for i in range(len(predictions)):
            prediction = predictions[i][0]
            solution = solutions[i]

            if (prediction > 0 and solution == 1.0):
                matrix['true_positive'] = matrix['true_positive'] + 1
            elif (prediction > 0 and solution == -1.0):
                matrix['false_positive'] = matrix['false_positive'] + 1
            elif (prediction < 0 and solution == -1.0):
                matrix['true_negative'] = matrix['true_negative'] + 1
            elif (prediction < 0 and solution == 1.0):
                matrix['false_negative'] = matrix['false_negative'] + 1
            else:
                raise Exception("unreachable code!")
        
        matrix['total'] = matrix['true_positive'] + matrix['true_negative'] + matrix['false_negative'] + matrix['false_positive']

        return matrix

            