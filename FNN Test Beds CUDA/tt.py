import csv
import matplotlib.pyplot as plt

def readFile(name):
    inp = []
    with open(name) as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        for row in reader: # each row is a list
            inp.append(row)
    return inp[0]

preds = readFile("Predictions.txt")
outs = readFile("Outputs.txt")
errors = readFile("DigitErrors.txt")


def plotData(outs, preds, errs):
    pl = []
    for i in range(0, len(outs)):
        pl.append([outs[i], preds[i], errs[i]])
    
    pl = sorted(pl)
    nerrs = sorted(errs)
    outs = [i[0] for i in pl]
    preds = [i[1] for i in pl]
    errs = [i[2] for i in pl]
    plt.figure(0)
    plt.plot(outs)
    plt.plot(preds)
    plt.figure(1)
    plt.plot(nerrs)
    plt.figure(2)
    plt.plot(errs)
    plt.figure(3)
    plt.plot(outs)
    plt.figure(4)
    plt.plot(preds)
    plt.show()

plotData(outs, preds, errors)