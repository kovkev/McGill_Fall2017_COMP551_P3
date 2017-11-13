import numpy as np
import time
import csv
from sklearn import datasets, linear_model

train_x = []
train_y = []
test_x = []
with open("train_x.csv") as csvfile:
    reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = np.array(row).astype(float).astype(np.uint8)
        row[row>240] = 255
        row[row<=240] = 0
        train_x.append(row)

with open("train_y.csv") as csvfile:
    reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = np.array(row).astype(float).astype(np.uint8)
        train_y.append(row)

with open("test_x.csv") as csvfile:
    reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
    for row in reader:
        row = np.array(row).astype(float).astype(np.uint8)
        row[row>240] = 255
        row[row<=240] = 0
        test_x.append(row)


regr = linear_model.LogisticRegression()

train_y = np.array(train_y)
train_y = train_y.ravel()
print "fitting the model"
print time.clock()
regr.fit(train_x[0:10000], train_y[0:10000])
print time.clock()
print "model training complete"

test_y = regr.predict(test_x)

file = open('test_y.csv', 'w')
file.write("Id,label\n")
counter = 0
for output in test_y:
    counter += 1
    file.write("%s,%s\n" % (counter, output))
file.close()
