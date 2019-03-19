import utility.run as run
from utility.enums import DataGenerationPattern, Processor, RnnType
import ensembles.ensemble as evaluation
import glob, os, csv, time, argparse, sys
import ensembles.pruning as pruningWrapper
import utility.dataoperations as dataoperations

import datadefinitions.cargo2000 as cargo2000


# Get arguments from commandline
parser = argparse.ArgumentParser()
parser.add_argument('--log_file', help="if true programm will log to a timestamped log file. Defaults to false", type=bool, default=False)
sys_args = parser.parse_args()

# variables for evaluation
pruning_accuracy_lower_bound = [0.1]
pruning_diversity_lower_bound = [0.1]
pruning_ensemble_sizes = [3, 50, 100, 150, 200, 250, 300]

train_args = {
    'datageneration_pattern': DataGenerationPattern.Fit,
    'datadefinition': cargo2000.Cargo2000(),
    'processor': Processor.GPU,
    'cudnn': True,
    #'bagging': True, 
    #'bagging_size': bagging_size, 
    'validationdata_split': 0.05, 
    'testdata_split': 0.3333,  
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
test_x, test_y = evaluation.prepare_data(args, 'testdata')
val_x, val_y = evaluation.prepare_data(args, 'validationdata')

if not os.path.exists('models'):
    exit()

base_path = os.getcwd()
os.chdir('models')

if sys_args.log_file:
    std_out = sys.stdout
    log_file = open('ensemble_evaluation.log', 'w')
    sys.stdout = log_file

print("Create results file...")
with open('ensemble_results.csv'.format(), 'w', newline='') as result_file:
    writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["type", "ensemble", "sequence_id", "accuracy_method" ,"accuracy_lower_bound", "diversity_method", "diversity_lower_bound", 
    "original_size", "pruned_size", "ensemble_prediction", "ensemble_prediction_binary", "ground_truth", "ground_truth_plannedtimestamp", "ground_truth_binary", "prefix", "suffix"])

print("Create accuracy measures file...")
with open('ensemble_accuracy_measurements.csv'.format(), 'w', newline='') as acc_file:
    writer = csv.writer(acc_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["type", "ensemble", "accuracy_method", "accuracy_lower_bound", "diversity_method", "diversity_lower_bound", "original_size", "pruned_size", 'precision', 'recall', 'specificity', 'false_positive_rate_norm', 'false_positive_rate', 'negative_prediction_value', 'accuracy', 'f1', 'mcc_norm', 'mcc'])

print("Running through ensembles types...")
for ensemble_type in glob.glob('*'):
    if not os.path.isdir(ensemble_type):
        print("{} is not a directory".format(ensemble_type))
        continue
    
    models_path = os.getcwd()
    os.chdir(ensemble_type)
    print('[Ensemble] Evaluating all {} ensembles...'.format(ensemble_type))
    ensemble_result_data = []
    ensemble = evaluation.GenericEnsembleWrapper()
    for models in glob.glob('*'):
        print(models)
        for max_size in pruning_ensemble_sizes:
            print('Do run for max {} models...'.format(max_size))
            ensemble.load_models(models, max_size, args)
            original_models = ensemble.models
            original_weights = ensemble.weights

            pruning_param_combinations = []
            no_prune_count = 0
            for accuracy_lower_bound in pruning_accuracy_lower_bound:
                for diversity_lower_bound in pruning_diversity_lower_bound:
                    for accuracy_method in pruningWrapper.AccuracyPruningMethods:
                        for diversity_method in pruningWrapper.DiversityPruningMethods:
                            if accuracy_lower_bound == 0.0 and diversity_lower_bound == 0.0 and no_prune_count > 0:
                                continue
                            
                            no_prune_count = 1
                            pruining_params = {
                                'a_l_b': accuracy_lower_bound,
                                'd_l_b': diversity_lower_bound,
                                'a_m': accuracy_method.value,
                                'd_m': diversity_method.value
                            }
                            pruning_param_combinations.append(pruining_params)
            
            for params in pruning_param_combinations:
                print('Do configuration:\n{}'.format(params))
                ensemble.models = original_models
                ensemble.weights = original_weights
                pruner = pruningWrapper.PruningWrapper(params, args)
                print("Start pruning...")
                pruned_data = pruner.do_pruning(ensemble.models, ensemble.weights, val_x, val_y, models)
                ensemble.models = pruned_data['models']
                ensemble.weights = pruned_data['weights']
                if len(ensemble.models) == 0:
                    print("Abort for configuration {}, because there are no more models.".format(params))
                    continue
                print("Start evaluation...")
                ensemble.evaluate(test_x, test_y, args, ensemble_type, models, params, len(original_models), len(pruned_data['models']))   
            
    os.chdir(models_path)

if sys_args.log_file:
    sys.stdout = std_out
    log_file.close()