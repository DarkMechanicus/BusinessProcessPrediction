import ensembles.ensemble as evaluation
import os, glob, csv
import utility.dataoperations as dataoperations
from utility.enums import DataGenerationPattern, Processor, RnnType
import utility.run as run
import ensembles.pruning as pruningWrapper

import datadefinitions.cargo2000 as cargo2000

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

ensemble = evaluation.GenericEnsembleWrapper()

base_path = os.getcwd()
os.chdir('models')

for ensemble_type in glob.glob('*'):
    if not os.path.isdir(ensemble_type):
        print("{} is not a directory".format(ensemble_type))
        continue
    
    print('[Ensemble] Precomputing all {} ensembles...'.format(ensemble_type))
    models_path = os.getcwd()
    os.chdir(ensemble_type)
    for models in glob.glob('*'):
        ensemble.load_models(models, 300, args)
        params_proto = {
            'a_l_b': 0.0,
            'd_l_b': 0.0,
            'd_m': pruningWrapper.DiversityPruningMethods.CorrelationCoefficient.value,
        }
        params = []
        for method in pruningWrapper.AccuracyPruningMethods:
            proto = params_proto.copy()
            proto['a_m'] = method.value
            params.append(proto)

        for i in range(len(ensemble.models)):
            with open('{}/{:03}-model_accuracy_measures.csv'.format(models, i), 'w', newline='') as result_file:
                    writer = csv.writer(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow(["accuracy-measure", "accuracy", "true_positive", "false_positive", "true_negative", "false_negative", "total"])

        for p in params:
            print("Precomputing method '{}' for {} models".format(p['a_m'], len(ensemble.models)))
            pruner = pruningWrapper.PruningWrapper(p, args)
            pruner.do_accuracy_measure_computation(ensemble.models, ensemble.weights, val_x, val_y, models)
    
    os.chdir(models_path)