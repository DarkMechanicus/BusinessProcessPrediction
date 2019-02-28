import ensembles.ensemble as utility
import utility.run as run
from utility.enums import DataGenerationPattern, Processor, RnnType
import argparse, sys

import datadefinitions.cargo2000 as cargo2000

path = utility.get_model_path('bagging')

# Get arguments from commandline
parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', help='maximum of epochs the models will be trained for. Defaults to 300', type=int, default=300)
parser.add_argument('--ensemble_size', help='amount of models that will be trained in this ensemble. Defaults to 500', type=int, default=500)
parser.add_argument('--bagging_size', help='size of the bag. Defaults to 1.0', type=float, default=1.0)
parser.add_argument('--log_file', help="if true programm will log to a timestamped log file. Defaults to false", type=bool, default=False)
sys_args = parser.parse_args()

if sys_args.log_file:
    std_out = sys.stdout
    log_file = open('{}/ensemble_training.log'.format(path), 'w')
    sys.stdout = log_file

train_args = {
    'datageneration_pattern': DataGenerationPattern.Fit,
    'datadefinition': cargo2000.Cargo2000(),
    'processor': Processor.GPU,
    'cudnn': True,
    'bagging': True, 
    'bagging_size': sys_args.bagging_size, 
    'validationdata_split': 0.05, 
    'testdata_split': 0.3333,  
    'max_sequencelength': 50, 
    'batch_size': 64,   
    'neurons': 100,  
    'dropout': 0,
    'max_epochs': sys_args.max_epochs,
    'layers': 2,
    'save_model': False,
    'adaboost': None
}

for i in range(sys_args.ensemble_size):
    print('[Bagging] Training Model {}...'.format(i+1))
    str_model_name = path + '/{:03}'.format(i)
    train_args['running'] = str_model_name
    run.Train_And_Evaluate(**train_args)
    print('[Bagging] Training Model {} done.'.format(i+1))

if sys_args.log_file:
    sys.stdout = std_out
    log_file.close()