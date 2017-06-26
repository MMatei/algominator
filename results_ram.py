import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from tasks.copy import MemoryTask
from tasks.module import AddTask, SubTask, AddTaskBaby
import numpy as np

class MultiLayerLSTM(nn.Module):
    def __init__(self, layers_no=2, hidden_size=128):
        super(MultiLayerLSTM, self).__init__()

        self.layers_no = layers_no
        self.hidden_size = hidden_size

        self.layers = layers = []
        self.layer0 = layer0 = nn.LSTMCell(22, hidden_size)
        self.layers.append(layer0)

        for l in range(1, layers_no):
            new_layer = nn.LSTMCell(hidden_size, hidden_size)
            setattr(self, "layer" + str(l), new_layer)
            self.layers.append(new_layer)

        self.output = nn.Linear(hidden_size, 10)

    def forward(self, x, hidden_state=None):
        hidden_size = self.hidden_size
        layers_no = self.layers_no
        batch_size = x.size(0)

        if hidden_state is None:
            hidden_state = []
            for l in range(layers_no):
                h = x.data.new().resize_(batch_size, hidden_size).zero_()
                c = x.data.new().resize_(batch_size, hidden_size).zero_()
                hidden_state.append((Variable(h), Variable(c)))

        new_hidden_state = []

        for l, (f, h) in enumerate(zip(self.layers, hidden_state)):
            x, c = f(x, h)
            new_hidden_state.append((x, c))

        y = F.log_softmax(self.output(x))
        return y, new_hidden_state

    def run(self, X, T):
        h, loss = None, None
        Y = []
        for s in range(0, X.size(0)):
            y, h = self.forward(X[s], h)
            Y.append(y.data.numpy())
            loss_t = F.nll_loss(y, T[s])
            loss = (loss + loss_t) if loss is not None else loss_t
        return Y, loss.data[0]

    def train_step(self, X, T, o):
        h, loss = None, None
        for s in range(0, X.size(0)):
            y, h = self.forward(X[s], h)
            loss_t = F.nll_loss(y, T[s])
            loss = (loss + loss_t) if loss is not None else loss_t
        o.zero_grad()
        loss.backward()
        o.step()
        return loss.data[0]

def testModelOn(testX, testY, task):
    y, loss = model.run(testX, testY)
    allCorrect, bitsCorrect = task.analyzeRez(y, testY.data.numpy())
    return loss, allCorrect, bitsCorrect

def threshPruning(params, thresh):
    for param in params:
        p = param.data.numpy()
        for i,row in enumerate(p):
            if not isinstance(row, np.float32):
                for j,col in enumerate(row):
                    if abs(p[i,j]) < thresh:
                        p[i,j] = 0
        param.data = torch.from_numpy(p)

def hardPruning(model, validX, validY, task):
    loss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
    for param in model.parameters():
        p = param.data.numpy()
        for i,row in enumerate(p):
            if isinstance(row, np.ndarray):
                for j,col in enumerate(row):
                    if abs(p[i][j]) <= 0.3: #combine with thresh pruning
                        p[i][j] = 0
                        continue
                    p[i][j] = 0
                    loss, ac, bc = testModelOn(validX, validY, task)
                    if ac >= allCorrect:
                        allCorrect = ac
                    else:
                        p[i][j] = col
        param.data = torch.from_numpy(p)

ts1 = time.time()
# task = MemoryTask()
task = SubTask()
model = MultiLayerLSTM(1, 64)
# optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=0.0005)
# optimizer = optim.Adam(model.parameters(), lr = 0.0001)
optimizer = optim.RMSprop(model.parameters())
BATCH_SIZE = 1000
TRAIN_EXPLES = 4000
PASSES = TRAIN_EXPLES/BATCH_SIZE

ts2 = time.time()
print("Initialization completed in " + str(ts2 - ts1) +" seconds.")

# trainX, trainY = task.getDataStress(25, TRAIN_EXPLES, True)
trainX, trainY = task.getData(10, TRAIN_EXPLES)
trainX = Variable(torch.from_numpy(trainX)).float()
trainY = Variable(torch.from_numpy(trainY)).long()
validX, validY = task.getData(15, 300)
validX = Variable(torch.from_numpy(validX)).float()
validY = Variable(torch.from_numpy(validY)).long()
testX, testY = task.getData(20, 300)
testX = Variable(torch.from_numpy(testX)).float()
testY = Variable(torch.from_numpy(testY)).long()
history = {'loss':[],'val_loss':[],'val_acc':[],'val_acce':[]}
ts1 = time.time()
print("Data generated in " + str(ts1 - ts2) +" seconds.")

for epoch in range(1,100):
    trainLoss = 0
    for i in range(0, TRAIN_EXPLES, BATCH_SIZE):
        # trainLoss += model.forward(trainX[:,i:i+BATCH_SIZE], trainY[:,i:i+BATCH_SIZE], optimizer)[1]
        trainLoss += model.train_step(trainX[:,i:i+BATCH_SIZE], trainY[:,i:i+BATCH_SIZE], optimizer)
        # optimizer.step()
    # if epoch % 20 == 0:
        # threshPruning(model.parameters(), 0.1)
        # hardPruning(model, validX, validY, task)
    validLoss, allCorrect, bitsCorrect = testModelOn(validX, validY, task)
    history['loss'].append(trainLoss/PASSES)
    history['val_loss'].append(validLoss)
    history['val_acc'].append(bitsCorrect*100)
    history['val_acce'].append(allCorrect*100)
    print(epoch)
    print(allCorrect)
ts2 = time.time()
print "Model trained in " + str(ts2 - ts1) +" seconds."

loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
print loss
print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

# for param in model.parameters():
#     # print(param.data)
#     p = param.data.numpy()
#     for i,row in enumerate(p):
#         if isinstance(row, np.ndarray):
#             for j,col in enumerate(row):
#                 if abs(p[i][j]) < 0.3:
#                     p[i][j] = 0
#             # loss, allCorrect, bc = testModelOn(validX, validY, task)
#             # if bc >= bitsCorrect:
#             #   bitsCorrect = bc
#             # else:
#             #   p[i][j] = col
#     # param.data = torch.from_numpy(param.data.numpy().round())
#     param.data = torch.from_numpy(p)
#     # print(param.data)
# loss, allCorrect, bitsCorrect = testModelOn(testX, testY, task)
# print loss
# print "Numbers correctly predicted: "+str(allCorrect*100)+"%"
# print "Bits correctly predicted: "+str(bitsCorrect*100)+"%"

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