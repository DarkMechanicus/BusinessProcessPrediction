import utility.run as run
from utility.enums import DataGenerationPattern, Processor, RnnType
import ensembles.ensemble as evaluation
import glob, os, csv, time
import ensembles.pruning as pruningWrapper
import utility.dataoperations as dataoperations

import datadefinitions.cargo2000 as cargo2000

# variables for evaluation
pruning_accuracy_lower_bound = [0.1]
pruning_diversity_lower_bound = [0.1]

train_args = {
    'datageneration_pattern': DataGenerationPattern.Fit,
    'datadefinition': cargo2000.Cargo2000(),
    'processor': Processor.GPU,
    'cudnn': True,
    #'bagging': True, 
    #'bagging_size': bagging_size, 
    'validationdata_split': 0.05, 
    'testdata_split': 0.05,  
    'max_sequencelength': 50, 
    'batch_size': 64,   
    'neurons': 100,  
    'dropout': 0,
    #'max_epochs': max_epochs,
    'layers': 2,
    #'save_model': False,
    #'adaboost': None
}

args = run.Do_Preprocessing(**train_args)
args['test_sentences'] = dataoperations.CreateSentences(args['testdata'])
test_x, test_y = evaluation.prepare_data(args)

if not os.path.exists('models'):
    exit()

base_path = os.getcwd()
os.chdir('models')
with open('../ensemble_results.csv'.format(), 'w', newline='') as result_file:
    writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["type", "ensemble", "sequence_id", "accuracy_method" ,"accuracy_lower_bound", "diversity_method", "diversity_lower_bound", "ensemble_prediction", "ensemble_prediction_binary", "ground_truth", "ground_truth_plannedtimestamp", "ground_truth_binary", "prefix", "suffix"])
    
with open('../ensemble_accuracy_measurements.csv'.format(), 'w', newline='') as acc_file:
    writer = csv.writer(acc_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["type", "ensemble", "accuracy_method", "accuracy_lower_bound", "diversity_method", "diversity_lower_bound", 'precision', 'recall', 'specificity', 'false_positive_rate', 'negative_prediction_value', 'accuracy', 'f1', 'mcc'])

for ensemble_type in glob.glob('*'):
    if not os.path.isdir(ensemble_type):
        continue
    
    models_path = os.getcwd()
    os.chdir(ensemble_type)
    print('[Ensemble] Evaluating all {} ensembles...'.format(ensemble_type))
    ensemble_result_data = []
    ensemble = evaluation.GenericEnsembleWrapper()
    for models in glob.glob('*'):
        ensemble.load_models(models, args)
        original_models = ensemble.models
        original_weights = ensemble.weights

        pruning_param_combinations = []
        for accuracy_lower_bound in pruning_accuracy_lower_bound:
            for diversity_lower_bound in pruning_diversity_lower_bound:
                for accuracy_method in pruningWrapper.AccuracyPruningMethods:
                    for diversity_method in pruningWrapper.DiversityPruningMethods:
                        pruining_params = {
                            'a_l_b': accuracy_lower_bound,
                            'd_l_b': diversity_lower_bound,
                            'a_m': accuracy_method.value,
                            'd_m': diversity_method.value
                        }
                        pruning_param_combinations.append(pruining_params)
        
        for params in pruning_param_combinations:
            loop_start = time.time()
            print(params)
            ensemble.models = original_models
            ensemble.weights = original_weights
            pruner = pruningWrapper.PruningWrapper(params, args)
            pruning_start = time.time()
            print("Pruning...")
            pruned_data = pruner.do_pruning(ensemble.models, ensemble.weights, test_x, test_y, models)
            ensemble.models = pruned_data['models']
            ensemble.weights = pruned_data['weights']
            print(time.time() - pruning_start)
            if len(ensemble.models) == 0:
                continue
            ensemble.evaluate(test_x, test_y, args, ensemble_type, models, params)
            print(time.time() - loop_start)

        ensemble.models = original_models
        ensemble.weights = original_weights
        ensemble.load_models(models, args)
        ensemble.evaluate(test_x, test_y, args, ensemble_type, models, {
            'a_m': 'None',
            'a_l_b': 0.0,
            'd_m': 'None',
            'd_l_b': 0.0
        })      
            
    os.chdir(models_path)
