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
		inData = np.random.binomial(1,0.5,size=(seqSz, batchSz, self.inputSz)) * 2 - 1
		target = []
		for i in range(0,batchSz):
			index = randint(0, seqSz-2)
			inData[0,i,:] = self.toBinary(index) * 2 - 1
			target.append(inData[index + 1,i,:])
		return (inData, np.array(target))

class MemoryTaskOld(Task):
	def __init__(self):
		self.inputSz = 21
		self.outputSz = 10

	def getData(self, seqSz, batchSz, zero=False):
		if seqSz < 2:
			print("Sequence size less than 2!")
			return None
		if zero:
			inData = np.zeros((seqSz, batchSz, self.inputSz))
			target = np.zeros((seqSz, batchSz))
		else:
			inData = - np.ones((seqSz, batchSz, self.inputSz))
			target = - np.ones((seqSz, batchSz, self.outputSz))
		writeSeq = min(10, seqSz / 2) # write at most once in every cell
		for b in range(0, batchSz):
			memory = [0] * 10
			for i in range(0, writeSeq):
				# TODO: don't write at consec locations
				r = randint(0,9)
				memory[i] = r
				inData[i,b,r] = 1
				inData[i,b,10+i] = 1 # address
				if zero:
					target[i,b] = r
				else:
					target[i,b,r] = 1
			for i in range(writeSeq, seqSz):
				a = randint(0,9)
				inData[i,b,10+a] = 1
				inData[i,b,20] = 1 # signal read op
				if zero:
					target[i,b] = memory[a]
				else:
					target[i,b,memory[a]] = 1
		return (inData, target)

	def getDataStress(self, seqSz, batchSz, zero=False):
		if seqSz < 2:
			print("Sequence size less than 2!")
			return None
		if zero:
			inData = - np.ones((seqSz, batchSz, self.inputSz))
			# inData = np.zeros((seqSz, batchSz, self.inputSz))
			target = np.zeros((seqSz, batchSz))
		else:
			inData = - np.ones((seqSz, batchSz, self.inputSz))
			target = - np.ones((seqSz, batchSz, self.outputSz))
		for b in range(0, batchSz):
			memory = [-1] * 10
			for i in range(0, seqSz):
				read = randint(0,1)
				addr = randint(0,9)
				if read == 0 or memory[addr] == -1:
					r = randint(0,9)
					memory[addr] = r
					inData[i,b,r] = 1
					inData[i,b,10+addr] = 1
					if zero:
						target[i,b] = r
					else:
						target[i,b,r] = 1
				else:
					inData[i,b,10+addr] = 1
					inData[i,b,20] = 1 # signal read op
					if zero:
						target[i,b] = memory[addr]
					else:
						target[i,b,memory[addr]] = 1
		return (inData, target)
	def analyzeRez(self, output, target, zero=False):
		correct = 0
		for s in range(0, target.shape[0]):
			for b in range(0, target.shape[1]):
				if zero:
					if output[s][b].argmax() == target[s,b]:
						correct += 1
				else:
					if output[s][b].argmax() == target[s,b].argmax():
						correct += 1
		if zero:
			return (correct / (target.size), 0.9)
		return (correct / (output.size/self.outputSz), 0.9)

class MemoryTask(Task):
	def __init__(self):
		self.inputSz = 22
		self.outputSz = 11

	def getDataStress(self, seqSz, batchSz, unused=True):
		if seqSz < 2:
			print("Sequence size less than 2!")
			return None
		inData = - np.ones((seqSz, batchSz, self.inputSz))
		# inData = np.zeros((seqSz, batchSz, self.inputSz))
		target = np.zeros((seqSz, batchSz))
		for b in range(0, batchSz):
			memory = [10] * 10
			for i in range(0, seqSz):
				read = randint(0,1)
				addr = randint(0,9)
				if read == 0:
					r = randint(0,9)
					memory[addr] = r
					inData[i,b,r] = 1
					inData[i,b,11+addr] = 1
					target[i,b] = r
				else:
					inData[i,b,10] = 1 # value is null
					inData[i,b,11+addr] = 1
					inData[i,b,20] = 1 # signal read op
					target[i,b] = memory[addr]
		return (inData, target)
	def analyzeRez(self, output, target):
		correct = 0
		total = 0
		for b in range(0, target.shape[1]):
			crct = 0
			for s in range(0, target.shape[0]):
				if output[s][b].argmax() == target[s,b]:
					crct += 1
			if crct == target.shape[0]:
				total += 1
			correct += crct
		return (correct / (target.size), total / target.shape[1])

class BinaryRAMTask(Task):
	def __init__(self):
		self.inputSz = 21
		self.outputSz = 10

	def getData(self, seqSz, batchSz):
		if seqSz < 2:
			print("Sequence size less than 2!")
			return None
		inData = - np.ones((seqSz, batchSz, self.inputSz))
		target = - np.ones((seqSz, batchSz, self.outputSz))
		for b in range(0, batchSz):
			memory = [-1] * 10
			for i in range(0, seqSz):
				read = randint(0,1)
				addr = randint(0,9)
				if read == 0 or memory[addr] == -1:
					r = randint(0,9)
					memory[addr] = r
					inData[i,b,r] = 1
					inData[i,b,10+addr] = 1
					target[i,b,:] = self.toBinary(r) * 2 - 1
				else:
					inData[i,b,10+addr] = 1
					inData[i,b,20] = 1 # signal read op
					target[i,b,:] = self.toBinary(memory[addr]) * 2 - 1
		return (inData, target)