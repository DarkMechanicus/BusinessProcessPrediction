import csv, os, math
import ensembles.accuracy_measures as acc

buckets = [
    [], # x <= 10%  0
    [], # x <= 20%  1
    [], # x <= 30%  2
    [], # x <= 40%  3
    [], # x <= 50%  4
    [], # x <= 60%  5
    [], # x <= 70%  6
    [], # x <= 80%  7
    [], # x <= 90%  8
    []  # x <= 100% 9
]

ensemble_accuracy = {}

halfway_mark = '13' # 50% process completion
midbucket_index = 4
prefix_index = 14
suffix_index = 15
binary_prediction_index = 10
binary_truth_index = 13

print('Bucketing results...')
with open('./models/ensemble_results.csv'.format(), 'r', newline='') as result_file:
    reader = csv.reader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    next(reader, None) # skip header
    for line in reader:
        sequence = line[prefix_index] + ' ' + line[suffix_index]
        sequence = sequence.replace(' !', '') #remove ! which stands for the end of the sequence
        sequence_list = sequence.split()
        prefix_list = line[prefix_index].replace(' !', '').split()
        suffix_list = line[suffix_index].replace(' !', '').split()
        
        sequence_length = len(sequence_list)
        halfway_index = sequence_list.index(halfway_mark)

        process_completion = (len(prefix_list) / sequence_length) * 100
        bucket_index = round(process_completion / 10)

        if bucket_index == midbucket_index and process_completion < 50.0:
            bucket_index -= 1
        
        if bucket_index == midbucket_index and process_completion > 50.0:
            bucket_index += 1

        if bucket_index > 9:
            bucket_index = 9

        buckets[bucket_index].append(line)

print('Processing buckets...')
for i, bucket in enumerate(buckets):
    for line in bucket:
        ensemble_designation = line[0] + '_' + line[1] + '_' + line[3] + '_' + line[4] + '_' + line[5] + '_' + line[6] + '_' + line[7] + '_' + line[8]
        t = ensemble_accuracy.get(ensemble_designation, None)
        if (not t == None) and (not t.get(i, None) == None):
            continue
        else:
            ensemble_bucket_data = list(filter(lambda fline: fline[0] == line[0] and fline[1] == line[1] and fline[3] == line[3] and fline[4] == line[4] and fline[5] == line[5] and fline[6] == line[6] and fline[7] == line[7] and fline[8] == line[8], bucket))
            confusion_matrix = {
                'tp': 0,
                'fp': 0,
                'tn': 0,
                'fn': 0,
                'total': 0
            }

            for data in ensemble_bucket_data:
                predicion = int(data[binary_prediction_index])
                if data[binary_prediction_index] == data[binary_truth_index]:
                    confusion_matrix['tp'] += (1 if predicion > 0 else 0)
                    confusion_matrix['tn'] += (1 if predicion < 0 else 0)
                else:
                    confusion_matrix['fp'] += (1 if predicion > 0 else 0)
                    confusion_matrix['fn'] += (1 if predicion < 0 else 0)
                confusion_matrix['total'] += 1
            
            acc_results = {
                'precision': acc.compute_precision(confusion_matrix),
                'recall': acc.compute_recall(confusion_matrix),
                'specificity': acc.compute_specificity(confusion_matrix),
                'false_positive_rate_norm': acc.compute_false_positive_rate(confusion_matrix),
                'false_positive_rate': acc.compute_false_positive_rate(confusion_matrix, False),
                'negative_prediction_value': acc.compute_negative_prediction_value(confusion_matrix),
                'accuracy': acc.compute_accuracy(confusion_matrix),
                'f1': acc.compute_f1(confusion_matrix),
                'mcc_norm': acc.compute_mcc(confusion_matrix),
                'mcc': acc.compute_mcc(confusion_matrix, False)
            }

            if ensemble_accuracy.get(ensemble_designation, None) == None:
                ensemble_accuracy[ensemble_designation] = {}
            ensemble_accuracy[ensemble_designation][i] = acc_results

print('Writting bucketed results...')
with open('./models/processed_ensemble_results.csv', 'w', newline='') as processed_file:
    writer = csv.writer(processed_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["ensemble_designation", "bucket", 'precision', 'recall', 'specificity', 'false_positive_rate_norm', 'false_positive_rate', 'negative_prediction_value', 'accuracy', 'f1', 'mcc_norm', 'mcc'])
    for ensemble in ensemble_accuracy:
        for bucket in ensemble_accuracy[ensemble]:
            writer.writerow([ensemble, bucket] + list(ensemble_accuracy[ensemble][bucket].values()))
