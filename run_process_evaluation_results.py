import csv, os, math
import ensembles.accuracy_measures as acc

buckets = [
    [], # Step 1
    [], # Step 2
    [], # Step 3
    [], # Step 4
    [], # Step 13
    [], # Step 14
    [], # Step 15
    [] # Step 16
]

ensemble_accuracy = {}

prefix_index = 14
suffix_index = 15
binary_prediction_index = 10
binary_truth_index = 13

print('Bucketing results...')
with open('./models/ensemble_results.csv'.format(), 'r', newline='') as result_file, open('./models/bucket_0.csv', 'w', newline='') as bucket_0, open('./models/bucket_1.csv', 'w', newline='') as bucket_1, open('./models/bucket_2.csv', 'w', newline='') as bucket_2, open('./models/bucket_3.csv', 'w', newline='') as bucket_3, open('./models/bucket_4.csv', 'w', newline='') as bucket_4, open('./models/bucket_5.csv', 'w', newline='') as bucket_5, open('./models/bucket_6.csv', 'w', newline='') as bucket_6, open('./models/bucket_7.csv', 'w', newline='') as bucket_7:
    reader = csv.reader(result_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writers = []
    writers.append(csv.writer(bucket_0, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))
    writers.append(csv.writer(bucket_1, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))
    writers.append(csv.writer(bucket_2, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))
    writers.append(csv.writer(bucket_3, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))
    writers.append(csv.writer(bucket_4, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))
    writers.append(csv.writer(bucket_5, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))
    writers.append(csv.writer(bucket_6, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))
    writers.append(csv.writer(bucket_7, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL))

    next(reader, None) # skip header
    for line in reader:
        sequence = line[prefix_index] + ' ' + line[suffix_index]
        sequence = sequence.replace(' !', '') #remove ! which stands for the end of the sequence
        sequence_list = sequence.split()
        suffix_list = line[suffix_index].replace(' !', '').split()

        bucket_index = -1
        if suffix_list[0] == '1':
            bucket_index = 0
        elif suffix_list[0] == '2':
            bucket_index = 1
        elif suffix_list[0] == '3':
            bucket_index = 2
        elif suffix_list[0] == '4':
            bucket_index = 3
        elif suffix_list[0] == '13':
            bucket_index = 4
        elif suffix_list[0] == '14':
            bucket_index = 5
        elif suffix_list[0] == '15':
            bucket_index = 6
        elif suffix_list[0] == '16':
            bucket_index = 7
        
        if bucket_index == -1:
            continue

        writers[bucket_index].writerow(line)

print('Processing buckets...')
for i in range(len(buckets)):
    all_data = []
    with open('./models/bucket_{}.csv'.format(i), 'r', newline='') as bucket:
        reader = csv.reader(bucket, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for line in reader:
            all_data.append(line)

    with open('./models/bucket_{}.csv'.format(i), 'r', newline='') as bucket:
        reader = csv.reader(bucket, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        data_copy = list(all_data)
        for line in reader:
            ensemble_designation = line[0] + '_' + line[1] + '_' + line[3] + '_' + line[4] + '_' + line[5] + '_' + line[6] + '_' + line[7] + '_' + line[8]
            t = ensemble_accuracy.get(ensemble_designation, None)
            if (not t == None) and (not t.get(i, None) == None):
                continue
            else:
                ensemble_bucket_data = list(filter(lambda fline: fline[0] == line[0] and fline[1] == line[1] and fline[3] == line[3] and fline[4] == line[4] and fline[5] == line[5] and fline[6] == line[6] and fline[7] == line[7] and fline[8] == line[8], data_copy))
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
