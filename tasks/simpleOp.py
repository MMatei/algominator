from __future__ import division
import numpy as np
from random import randint
from task import NoSeqTask

# Given two numbers, compute their sum (including carry)
class SumTask(NoSeqTask):
	def __init__(self, valLen):
		self.valLen = valLen
		self.inputSz = 2 * valLen
		self.outputSz = valLen + 1
		self.maxSeq = 1

	def getData(self, seqSz, batchSz):
		if seqSz > self.valLen:
			raise Exception('seqSz exceeds max value length', str(seqSz)+' > '+str(self.valLen))

		inData = np.zeros((batchSz, self.inputSz))
		target = np.zeros((batchSz, self.outputSz))
		maxVal = pow(2, seqSz) - 1
		vl = self.valLen - 1
		for i in range(0, batchSz):
			a = randint(1, maxVal)
			b = randint(1, maxVal)
			c = a + b
			inData[i,:self.valLen] = self.toBinary(a)[self.valLen:]
			inData[i,self.valLen:] = self.toBinary(b)[self.valLen:]
			target[i,:] = self.toBinary(c)[vl:] # output 1 bit longer
		return (inData.reshape((batchSz,1,self.inputSz)), target)

# Given two numbers, and a 0/1 signal, add them on a 0, subtract them otherwise
class ChoiceTask(NoSeqTask):
	def __init__(self, valLen):
		self.valLen = valLen
		self.inputSz = 2 * valLen + 1
		# output has first bit reserved for +/- sign
		self.outputSz = valLen + 2
		self.maxSeq = 1

	def getData(self, seqSz, batchSz):
		if seqSz > self.valLen:
			raise Exception('seqSz exceeds max value length', str(seqSz)+' > '+str(self.valLen))

		inData = np.zeros((batchSz, self.inputSz))
		target = np.zeros((batchSz, self.outputSz))
		maxVal = pow(2, seqSz) - 1
		valLen1 = self.valLen + 1
		for i in range(0, batchSz):
			a = randint(1, maxVal)
			b = randint(1, maxVal)
			choice = randint(0, 1)
			if choice == 0:
				c = a + b
			else:
				c = a - b
			inData[i,0] = choice
			inData[i,1:valLen1] = self.toBinary(a)[valLen1:]
			inData[i,valLen1:] = self.toBinary(b)[valLen1:]
			if c >= 0:
				target[i,1:] = self.toBinary(c)[self.valLen:] # output 1 bit longer
			else:
				target[i,0] = 1 # minus sign
				target[i,1:] = self.toBinary(-c)[self.valLen:]
		return (inData.reshape((batchSz,1,self.inputSz)), target)