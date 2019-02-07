import ensembles.ensemble as utility
import utility.run as run
from utility.enums import DataGenerationPattern, Processor, RnnType

import datadefinitions.cargo2000 as cargo2000

path = utility.get_model_path('bagging')

# Bagging variables
max_epochs = 10
ensemble_size = 5
bagging_size = 1.0

train_args = {
    'datageneration_pattern': DataGenerationPattern.Fit,
    'datadefinition': cargo2000.Cargo2000(),
    'processor': Processor.GPU,
    'cudnn': True,
    'bagging': True, 
    'bagging_size': bagging_size, 
    'validationdata_split': 0.05, 
    'testdata_split': 0.05,  
    'max_sequencelength': 50, 
    'batch_size': 64,   
    'neurons': 100,  
    'dropout': 0,
    'max_epochs': max_epochs,
    'layers': 2,
    'save_model': False,
    'adaboost': None
}

for i in range(ensemble_size):
    print('[Bagging] Training Model {}...'.format(i+1))
    str_model_name = path + '/{:03}'.format(i)
    train_args['running'] = str_model_name
    run.Train_And_Evaluate(**train_args)
    print('[Bagging] Training Model {} done.'.format(i+1))