import matplotlib.pyplot as plot
import csv, os

ensembles = {}

x_axis = ['1','2','3','4','13','14','15','16']

colors = [
    '#FF0000',
    '#00FF00',
    '#0000FF',
    '#FFFF00',
    '#FF00FF',
    '#00FFFF',
    '#CCCCCC',
    '#000000'
]

if not os.path.exists('./plots'):
    os.mkdir('./plots')

with open('./models/processed_ensemble_results.csv', 'r', newline='') as data_file:
    reader = csv.reader(data_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    next(reader, None) # skip header
    for line in reader:
        if ensembles.get(line[0], None) == None:
            ensembles[line[0]] = {}
        ensembles[line[0]][line[1]] = line

for ensemble in ensembles:
    e = ensembles[ensemble]
    acc_measures = {
        'mcc_data': [],
        'f1_data' : [],
        'accuracy_data' : [],
        'np_data' : [],
        'fpr_data' : [],
        'specificity_data' : [],
        'recall_data' : [],
        'precision_data' : []
    }

    for bucket in e:
        line = e[bucket]
        acc_measures['mcc_data'].append(float(line[11]))
        acc_measures['f1_data'].append(float(line[9]))
        acc_measures['accuracy_data'].append(float(line[8]))
        acc_measures['np_data'].append(float(line[7]))
        acc_measures['fpr_data'].append(float(line[6]))
        acc_measures['specificity_data'].append(float(line[4]))
        acc_measures['recall_data'].append(float(line[3]))
        acc_measures['precision_data'].append(float(line[2]))

    plot.plot(x_axis, acc_measures['mcc_data'], colors[7], label='mcc')
    plot.plot(x_axis, acc_measures['f1_data'], colors[6], label='f1')
    plot.plot(x_axis, acc_measures['accuracy_data'], colors[5], label='accuracy')
    plot.plot(x_axis, acc_measures['np_data'], colors[4], label='np')
    plot.plot(x_axis, acc_measures['fpr_data'], colors[3], label='fpr')
    plot.plot(x_axis, acc_measures['specificity_data'], colors[2], label='specificity')
    plot.plot(x_axis, acc_measures['recall_data'], colors[1], label='recall')
    plot.plot(x_axis, acc_measures['precision_data'], colors[0], label='precision')

    plot.ylabel('accuracy measure')
    plot.xlabel('process step')
    plot.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plot.grid(True)
    plot.axis([0.0, 7.0, 0.0, 1.0])
    plot.savefig('./plots/{}.svg'.format(ensemble), transparent=True)
    plot.clf()

acc_measures = {
    'mcc_data': 11,
    'f1_data' : 9,
    'accuracy_data' : 8,
    'np_data' : 7,
    'fpr_data' : 6,
    'specificity_data' : 4,
    'recall_data' : 3,
    'precision_data' : 2
}

for am in acc_measures:
    for ensemble in ensembles:
        e = ensembles[ensemble]

        plot_data = []
        for bucket in e:
            line = e[bucket]
            plot_data.append(float(line[acc_measures[am]]))

        plot.plot(x_axis, plot_data, label=ensemble)

    plot.ylabel(am)
    plot.xlabel('process step')
    plot.grid(True)
    plot.axis([0.0, 7.0, 0.0, 1.0])
    plot.savefig('./plots/all_{}.svg'.format(am), transparent=True)