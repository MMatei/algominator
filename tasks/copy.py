import numpy as np
from keras.preprocessing.sequence import pad_sequences

# Copy the input to the output, without modification
class CopyTask:
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.binomial(1,0.5,size=(batchSz, seqSz, self.inputSz))
		inData = pad_sequences(inData, maxlen=self.maxSeq, padding='post')
		return (inData, inData.copy())

# Given a sequence of values, always write the first value in the sequence
class CopyFirstTask:
	def __init__(self, inputSz, maxSeq):
		self.inputSz = inputSz
		self.maxSeq = maxSeq

	def getData(self, seqSz, batchSz):
		inData = np.random.binomial(1,0.5,size=(batchSz, seqSz, self.inputSz))
		target = []
		for seq in inData:
			target.append([seq[0]] * seqSz)
		inData = pad_sequences(inData, maxlen=self.maxSeq, padding='post')
		target = pad_sequences(np.array(target), maxlen=self.maxSeq, padding='post')
		return (inData, target)