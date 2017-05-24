from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import time
from models.rnn import RNN
from models.sequential import Sequential
from pso_rnn import PSORNN, BPSORNN, GPSORNN
from chaos import ChaosPSORNN

from tasks.copy import CopyTask, CopyFirstTask, IndexTask
from tasks.algo import GCDTask
from tasks.simpleOp import SumTask, ChoiceTask
from tasks.signal import TimeSignal

def testModelOn(testX, testY, task):
	output = []
	for X in testX:
		model.reset(testX.shape[2]) # set to compute whole batch at once
		Y = model.forward(X)
		output.append(Y)
	output = np.array(output)
	loss = np.square(output - testY).sum() / testY.size
	allCorrect, bitsCorrect = task.analyzeRez(output, testY)
	return loss, allCorrect, bitsCorrect

ts1 = time.time()
batchSz = 30
task = CopyTask(8, 10)
model = Sequential()
model.add(RNN(8, 8, batchSz))
model.add(RNN(8, 8, batchSz))
model.add(RNN(8, 8, batchSz))
model.add(RNN(8, 8, batchSz))
# task = TimeSignal(15)
# model = Sequential()
# model.add(RNN(2, 32, batchSz))
# model.add(RNN(32,32,batchSz))
# model.add(RNN(32,32,batchSz))
# model.add(RNN(32, 1, batchSz))
trainer = PSORNN(model, task)
testX, testY = task.getData(10, 2000)

ts2 = time.time()
print "Initialization completed in " + str(ts2 - ts1) +" seconds."

trainer.train(200)
ts1 = time.time()
print "Model trained in " + str(ts1 - ts2) +" seconds."

loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
print "Test loss: "+str(loss)
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"