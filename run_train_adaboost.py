import ensembles.ensemble as utility
import utility.run as run
from utility.enums import DataGenerationPattern, Processor, RnnType
import ensembles.adaboost as wrapper

import datadefinitions.cargo2000 as cargo2000

path = utility.get_model_path('adaboost')

# Bagging variables
max_epochs = 10
ensemble_size = 5
resampling = False
resampling_size = 100
can_decrease_weights = False
modify_validation = False

adaboost = wrapper.AdaBoostWrapper(resampling, resampling_size, can_decrease_weights, modify_validation)

train_args = {
    'datageneration_pattern': DataGenerationPattern.Fit,
    'datadefinition': cargo2000.Cargo2000(),
    'processor': Processor.GPU,
    'cudnn': True,
    'bagging': False, 
    'bagging_size': 0.0, 
    'validationdata_split': 0.05, 
    'testdata_split': 0.05,  
    'max_sequencelength': 50, 
    'batch_size': 64,   
    'neurons': 100,  
    'dropout': 0,
    'max_epochs': max_epochs,
    'layers': 2,
    'save_model': False,
    'adaboost': adaboost
}

for i in range(ensemble_size):
    print('[AdaBoost] Training Model {:03}...'.format(i+1))
    str_model_name = path + '/{:03}'.format(i)
    train_args['running'] = str_model_name
    run.Train_And_Evaluate(**train_args)
    print('[AdaBoost] Training Model {:03} done.'.format(i+1))

adaboost.save_ensemble_weights(path)