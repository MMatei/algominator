import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from tasks.copy import MemoryTask
from tasks.module import AddTask, SubTask, MulTask
from models.composite import MultiLayerLSTM, Composite
import numpy as np

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
task = MulTask()
# model = MultiLayerLSTM(22, 11, 1, 110)
model = MultiLayerLSTM(22, 10, 1, 32)
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

for epoch in range(1,205):
    trainLoss = 0
    for i in range(0, TRAIN_EXPLES, BATCH_SIZE):
        trainLoss += model.train_step(trainX[:,i:i+BATCH_SIZE], trainY[:,i:i+BATCH_SIZE], optimizer)
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

torch.save(model, "model_inmultire.bin")
model = torch.load("model_inmultire.bin")

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