import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from tasks.copy import CopyTask, CopyFirstTask, IndexTask
import numpy as np

class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ninp, nhid, nlayers):
		super(RNNModel, self).__init__()
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, bias=False)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, bias=False)

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def forward(self, input, hidden):
		return self.rnn(input, hidden)

	def loss(self, t, o):
		return (t - o).pow(2).sum() / (t.size(0) * t.size(1) * t.size(2))
		t = (t + 1) / 2
		o = (o + 1) / 2
		perNumber = -(t * (o+0.00001).log() + (1 - t) * (1.00001 - o).log()).sum(dim=2) / t.size(2)
		return perNumber.sum() / (t.size(0) * t.size(1))

	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
					Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
		else:
			return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


def testModelOn(testX, testY, task):
	batchSz = testX.size()[1]
	h = model.init_hidden(batchSz)
	y, h = model(testX, h)
	loss = model.loss(testY, y).data[0]
	allCorrect, bitsCorrect = task.analyzeRez(y.data.numpy(), testY.data.numpy())
	return loss, allCorrect, bitsCorrect

ts1 = time.time()
task = CopyTask(8, 20)
# model = RNNModel('RNN_TANH', 8, 8, 2)
model = RNNModel('LSTM', 8, 8, 1)
optimizer = optim.RMSprop(model.parameters()) #Adam(model.parameters(), lr = 0.0001)
BATCH_SIZE = 100
TRAIN_EXPLES = 3000
PASSES = TRAIN_EXPLES/BATCH_SIZE

ts2 = time.time()
print("Initialization completed in " + str(ts2 - ts1) +" seconds.")

trainX, trainY = task.getData(12, TRAIN_EXPLES)
trainX = Variable(torch.from_numpy(trainX)).float()
trainY = Variable(torch.from_numpy(trainY)).float()
validX, validY = task.getData(16, 300)
validX = Variable(torch.from_numpy(validX)).float()
validY = Variable(torch.from_numpy(validY)).float()
testX, testY = task.getData(16, 300)
testX = Variable(torch.from_numpy(testX)).float()
testY = Variable(torch.from_numpy(testY)).float()
history = {'loss':[],'val_loss':[],'val_acc':[],'val_acce':[]}
ts1 = time.time()
print("Data generated in " + str(ts1 - ts2) +" seconds.")

for epoch in range(0,100):
	trainLoss = 0
	for i in range(0, TRAIN_EXPLES, BATCH_SIZE):
		h = model.init_hidden(BATCH_SIZE)
		y, h = model(trainX[:,i:i+BATCH_SIZE], h)
		loss = model.loss(trainY[:,i:i+BATCH_SIZE], y)
		trainLoss += loss.data[0]
		loss.backward()
		optimizer.step()
	validLoss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
	history['loss'].append(trainLoss/PASSES)
	history['val_loss'].append(validLoss)
	history['val_acc'].append(bitsCorrect*100)
	history['val_acce'].append(allCorrect*100)
ts2 = time.time()
print "Model trained in " + str(ts2 - ts1) +" seconds."

loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
print loss
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

for param in model.parameters():
	# print(param.data)
	p = param.data.numpy()
	for i,row in enumerate(p):
		for j,col in enumerate(row):
			p[i][j] = 0
			loss, allCorrect, bc = testModelOn(validX, validY, task)
			if bc >= bitsCorrect:
				bitsCorrect = bc
			else:
				p[i][j] = col
	# param.data = torch.from_numpy(param.data.numpy().round())
	param.data = torch.from_numpy(p)
	# print(param.data)
loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
print loss
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

# summarize history for accuracy
plt.figure(1)
plt.plot(history['val_acc'])
plt.plot(history['val_acce'])
plt.title('validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['% numbers correct', '% bits correct'], loc='upper left')
axes = plt.gca()
axes.set_ylim([0,100])

# summarize history for loss
plt.figure(2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()