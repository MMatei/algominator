from __future__ import division
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from random import randint
from task import Task, NoSeqTask

# Copy the input to the output, without modification
class CopyTask(Task):
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.outputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.binomial(1,0.5,size=(batchSz, seqSz, self.inputSz))
		inData = pad_sequences(inData, maxlen=self.maxSeq, padding='post')
		return (inData, inData.copy())

# Given a sequence of values, always write the first value in the sequence
class CopyFirstTask(Task):
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.outputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.binomial(1,0.5,size=(batchSz, seqSz, self.inputSz))
		target = []
		for seq in inData:
			target.append([seq[0]] * seqSz)
		inData = pad_sequences(inData, maxlen=self.maxSeq, padding='post')
		target = pad_sequences(np.array(target), maxlen=self.maxSeq, padding='post')
		return (inData, target)

# Given a sequence of values, the first of which is an index, write sequence[index]
class IndexTask(NoSeqTask):
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.outputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.binomial(1,0.5,size=(batchSz, seqSz, self.inputSz))
		target = []
		for seq in inData:
			index = randint(0, seqSz-2)
			seq[0] = self.toBinary(index)
			target.append(seq[index + 1])
		inData = pad_sequences(inData, maxlen=self.maxSeq, padding='post')
		return (inData, np.array(target))