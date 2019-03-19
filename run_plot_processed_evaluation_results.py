import matplotlib.pyplot as plot
import csv, os

ensembles = {}

x_axis = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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
    mcc_data = []
    for bucket in e:
        line = e[bucket]
        mcc_data.append(float(line[11]))

    plot.plot(x_axis, mcc_data, label=ensemble)
    plot.ylabel('MCC')
    plot.xlabel('process completion')
    plot.legend()
    plot.grid(True)
    plot.axis([0.0, 1.0, 0.0, 1.0])
    plot.savefig('./plots/{}.svg'.format(ensemble), transparent=True)
    plot.clf()

for ensemble in ensembles:
    e = ensembles[ensemble]
    mcc_data = []
    for bucket in e:
        line = e[bucket]
        mcc_data.append(float(line[11]))

    plot.plot(x_axis, mcc_data, label=ensemble)

plot.ylabel('MCC')
plot.xlabel('process completion')
plot.legend()
plot.grid(True)
plot.axis([0.0, 1.0, 0.0, 1.0])
plot.savefig('./plots/{}.svg'.format('test'), transparent=True)