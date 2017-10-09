from __future__ import division
import numpy as np
from random import randint
from task import Task
import torch
from torch.autograd import Variable

class MTask(Task):
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

class AddTask(MTask):
	def __init__(self):
		self.inputSz = 22
		self.outputSz = 10

	def getData(self, seqSz, batchSz):
		if seqSz < 2:
			print("Sequence size less than 2!")
			return None
		inData = - np.ones((seqSz, batchSz, self.inputSz))
		target = np.zeros((seqSz, batchSz))
		for b in range(0, batchSz):
			carry = 0
			for s in range(0, seqSz-1):
				x1 = randint(0,9)
				x2 = randint(0,9)
				inData[s,b,x1] = 1
				inData[s,b,11+x2] = 1
				y = x1 + x2 + carry
				carry = y / 10
				y = y % 10
				target[s,b] = y
			inData[seqSz-1,b,10] = 1
			inData[seqSz-1,b,21] = 1
			target[seqSz-1,b] = carry
		return inData, target

class SubTask(MTask):
	def __init__(self):
		self.inputSz = 22
		self.outputSz = 10

	def getData(self, seqSz, batchSz):
		if seqSz < 2:
			print("Sequence size less than 2!")
			return None
		inData = - np.ones((seqSz, batchSz, self.inputSz))
		target = np.zeros((seqSz, batchSz))
		for b in range(0, batchSz):
			carry = 0
			for s in range(0, seqSz-1):
				x1 = randint(0,9)
				x2 = randint(0,9)
				inData[s,b,x1] = 1
				inData[s,b,11+x2] = 1
				y = x1 - x2 - carry
				carry = 0 # carry got used
				if y < 0:
					carry = 1
					y = 10 + y
				target[s,b] = y
			inData[seqSz-1,b,10] = 1
			inData[seqSz-1,b,21] = 1
			target[seqSz-1,b] = carry
		return inData, target

class MulTask(MTask):
	def __init__(self):
		self.inputSz = 22
		self.outputSz = 10

	def getData(self, seqSz, batchSz):
		if seqSz < 2:
			print("Sequence size less than 2!")
			return None
		inData = - np.ones((seqSz, batchSz, self.inputSz))
		target = np.zeros((seqSz, batchSz))
		for b in range(0, batchSz):
			carry = 0
			for s in range(0, seqSz-1):
				x1 = randint(0,9)
				x2 = randint(0,9)
				inData[s,b,x1] = 1
				inData[s,b,11+x2] = 1
				y = x1 * x2 + carry
				carry = y / 10
				y = y % 10
				target[s,b] = y
			inData[seqSz-1,b,10] = 1
			inData[seqSz-1,b,21] = 1
			target[seqSz-1,b] = carry
		return inData, target