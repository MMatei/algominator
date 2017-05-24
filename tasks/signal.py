import numpy as np
from random import random
from task import Task
from math import sin, cos

class TimeSignal(Task):
	def __init__(self, maxSeq):
		self.inputSz = 2
		self.outputSz = 1 # next elem in sequence
		self.maxSeq = maxSeq

	def fun(self, seq):
		return cos(seq[0]) * seq[1] - seq[2] * seq[2] * sin(seq[3]) + 0.17 * seq[4]

	def getData(self, seqSz, batchSz):
		if seqSz < 6:
			print "ERROR: sequence size less than 6!"
			return None
		inData = np.zeros((seqSz-5, 2, batchSz))
		target = np.zeros((seqSz-5, 1, batchSz))
		for b in range(0, batchSz):
			seq = [0, 0, 0, random()*2 - 1, random()*2 - 1]
			T = []
			for j in range(5, seqSz):
				r = self.fun(seq[j-5:j])
				seq.append(r)
				target[j-5,0,b] = r
			for j in range(3,len(seq)-2):
				inData[j-3,:,b] = seq[j:j+2]
		return inData, target

	# AnalyzeRez has no meaning for this problem...
	def analyzeRez(self, output, target):
		return (0,0)
