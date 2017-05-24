from __future__ import division
import numpy as np
from random import randint
from task import Task, NoSeqTask

# Copy the input to the output, without modification
class CopyTask(Task):
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.outputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.randint(2, size=(seqSz, batchSz, self.inputSz)) * 2 - 1
		return (inData, inData.copy())

# Given a sequence of values, always write the first value in the sequence
class CopyFirstTask(Task):
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.outputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.randint(2, size=(seqSz, batchSz, self.inputSz)) * 2 - 1
		target = inData.copy()
		for i in range(1, seqSz):
			target[i,:,:] = target[0,:,:]
		return (inData, target)

# Given a sequence of values, the first of which is an index, write sequence[index]
class IndexTask(NoSeqTask):
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.outputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.binomial(1,0.5,size=(seqSz, self.inputSz, batchSz))
		target = []
		for i in range(0,batchSz):
			index = randint(0, seqSz-2)
			inData[0,:,i] = self.toBinary(index)
			target.append(seq[index + 1])
		return (inData, np.array(target))