import ensembles.ensemble as utility
import utility.run as run
from utility.enums import DataGenerationPattern, Processor, RnnType
import ensembles.adaboost as wrapper
import argparse, sys

import datadefinitions.cargo2000 as cargo2000

path = utility.get_model_path('adaboost')

# Get arguments from commandline
parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', help='maximum of epochs the models will be trained for. Defaults to 300', type=int, default=300)
parser.add_argument('--ensemble_size', help='amount of models that will be trained in this ensemble. Defaults to 500', type=int, default=500)
parser.add_argument('--resampling', help='if true resampling will be used. Defaults to false', type=bool, default=False)
parser.add_argument('--decrease_weights', help="if true weights will be decreased on successful predictions. Defauls to false", type=bool, default=False)
parser.add_argument('--modify_validation', help="if true validation data will be evaluated and weighted in the same way the trainings data is. Defaults to false", type=bool, default=False)
parser.add_argument('--log_file', help="if true programm will log to a timestamped log file. Defaults to false", type=bool, default=False)
sys_args = parser.parse_args()

if sys_args.log_file:
    std_out = sys.stdout
    log_file = open('{}/ensemble_training.log'.format(path), 'w')
    sys.stdout = log_file

# Bagging variables

adaboost = wrapper.AdaBoostWrapper(sys_args.resampling, sys_args.decrease_weights, sys_args.modify_validation)

train_args = {
    'datageneration_pattern': DataGenerationPattern.Fit,
    'datadefinition': cargo2000.Cargo2000(),
    'processor': Processor.GPU,
    'cudnn': True,
    'bagging': False, 
    'bagging_size': 0.0, 
    'validationdata_split': 0.05, 
    'testdata_split': 0.3333,  
    'max_sequencelength': 50, 
    'batch_size': 64,   
    'neurons': 100,  
    'dropout': 0,
    'max_epochs': sys_args.max_epochs,
    'layers': 2,
    'save_model': False,
    'adaboost': adaboost
}

for i in range(sys_args.ensemble_size):
    print('[AdaBoost] Training Model {:03}...'.format(i+1))
    str_model_name = path + '/{:03}'.format(i)
    train_args['running'] = str_model_name
    run.Train_Only(**train_args)
    print('[AdaBoost] Training Model {:03} done.'.format(i+1))

adaboost.save_ensemble_weights(path)

if sys_args.log_file:
    sys.stdout = std_out
    log_file.close()