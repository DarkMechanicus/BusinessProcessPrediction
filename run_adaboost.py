import utility.run as run
import os, glob, sys, datetime, csv
from utility.enums import DataGenerationPattern, Processor, RnnType
import adaboost.adaboost as adaboostWrapper
import utility.models as models
import pruning.pruning as pruningWrapper

# datasets to test
import datadefinitions.cargo2000 as cargo2000

sys_arv_len = len(sys.argv)
size = int(sys.argv[1]) if sys_arv_len > 1 else 11
epochs = int(sys.argv[2]) if sys_arv_len > 2 else 30
resampling = (int(sys.argv[3]) == 1) if sys_arv_len > 3 else False
pruning = (int(sys.argv[4]) == 1) if sys_arv_len > 4 else False
pruning_accuracy = float(sys.argv[5]) if sys_arv_len > 5 else 0.0
pruning_diversity = float(sys.argv[6]) if sys_arv_len > 6 else 0.0
pruning_method = pruningWrapper.DiversityPruningMethods(sys.argv[7]) if sys_arv_len > 7 else pruningWrapper.DiversityPruningMethods.DisagreementMeasure
evaluate = (int(sys.argv[8]) == 1) if sys_arv_len > 8 else True
evaluate_models = sys.argv[9] if sys_arv_len > 9 else ''

print('running with arguments:')
print('ensemble_size = {}\nepochs = {}\nresampling = {}'.format(size, epochs, resampling))
print('pruning = {}\nmin acc = {}\nmin diversity = {}\nmethod = {}'.format(pruning, pruning_accuracy, pruning_diversity, pruning_method))
print('evalute = {}\nmodels = {}'.format(evaluate, evaluate_models))

print('[AdaBoot] Setup model directory...')
datestr = datetime.datetime.today().strftime('%Y%m%d-%H%M%S')
if not os.path.exists('models'):
    os.mkdir('models')

if size > 0 and not os.path.exists('models/{}'.format(datestr)):
    os.mkdir('models/{}'.format(datestr))
if evaluate_models == '':
    evaluate_models = datestr

adaboostWrapper = adaboostWrapper.AdaBoostWrapper(resampling=resampling)
pruningWrapper = pruningWrapper.PruningWrapper(pruning_accuracy, pruning_diversity, pruning_method)
dataset = cargo2000.Cargo2000()

train_args = {
    'datageneration_pattern': DataGenerationPattern.Fit,
    'datadefinition': dataset,
    'processor': Processor.GPU,
    'cudnn': True,
    'bagging': False, 
    'bagging_size': 0, 
    'validationdata_split': 0.05, 
    'testdata_split': 0.05,  
    'max_sequencelength': 50, 
    'batch_size': 64,   
    'neurons': 10,  
    'dropout': 0,
    'max_epochs': epochs,
    'layers': 1,
    'save_model': False,
    'adaboost': adaboostWrapper
}

for i in range(1, size+1):
    print('starting training round {} / {}'.format(i, size))
    str_running = 'models/{}/{:03}'.format(datestr, i)
    train_args['running'] = str_running
    run.Train_And_Evaluate(**train_args)
    print('finished training round {} / {}'.format(i, size))

#save ensemble weights
if size > 0:
    adaboostWrapper.save_weights(datestr)

if evaluate:
    if adaboostWrapper.args is None:
        adaboostWrapper.args = run.Do_Preprocessing(**train_args)
        print('Preprocessing arguments')

    args = adaboostWrapper.args
    #prepare testdata
    print('Preparing test data...')
    test_x = []
    test_y = []
    for i in range(len(args['testdata'][0])):
        sequencelength = len(args['testdata'][0][i]) - 1 #minus eol character
        for prefix_size in range(1,sequencelength):   
            cropped_data = []
            for a in range(len(args['testdata'])):
                cropped_data.append(args['testdata'][a][i][:prefix_size])  

        ground_truth = args['testdata'][6][i][0] + args['offsets'][6] #undo offset
        ground_truth_plannedtimestamp = args['testdata'][7][i][0] + args['offsets'][7] #undo offset
        prepared_data = dataset.EncodePrediction(cropped_data, args)
        prepared_truth = -1 if ground_truth <= ground_truth_plannedtimestamp else 1
        test_x.append(prepared_data)
        test_y.append(prepared_truth)

    #load ensemble from disc
    adaboostWrapper.load_ensemble(evaluate_models)

    if pruning:
        pruned = pruningWrapper.do_pruning(adaboostWrapper.ensemble_models, adaboostWrapper.ensemble_weights, test_x, test_y)
        adaboostWrapper.ensemble_weights = pruned['weights']
        adaboostWrapper.ensemble_models = pruned['models']

    if len(adaboostWrapper.ensemble_models) == 0:
        print('All Models have been pruned!')
        exit()

    print('Evaluate AdaBoost Ensemble of size: {}'.format(len(adaboostWrapper.ensemble_models)))    
    
    truth_matrix = []    
    for i in range(len(test_x)):
        test_data = test_x[i]
        test_solution = test_y[i]
        ensemble_prediction = adaboostWrapper.predict(test_data)
        print('Ensemble prediction: {} Ground Truth: {}'.format(ensemble_prediction, test_solution))
        truth_matrix.append({'prediction': ensemble_prediction[0][0], 'ground_truth': test_solution})

    correct_classifications = 0
    for classification in truth_matrix:
        if (classification['prediction'] > 0 and classification['ground_truth'] == 1) or (classification['prediction'] < 0 and classification['ground_truth'] == -1):
            correct_classifications += 1

    print('For {} test data sets this ensemble was right in {} cases ({} %)'.format(len(args['testdata'][0]), correct_classifications, (correct_classifications / len(args['testdata'][0]) * 100)))

    #save ensemble results
    adaboostWrapper.save_result(evaluate_models, truth_matrix)
        