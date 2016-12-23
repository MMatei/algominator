import numpy as np
import random
import math

class Particle:
	VMIN = -0.01
	VMAX = 0.01
	C1 = 0.6 # TODO: investigate C1, C2
	C2 = 1.2
	W0 = 1.8 # W's are given by PSO-BP paper
	W1 = 0.6
	LINGEN = 100 # nr of generations in which w progresses linearly
	K = 100 # the bigger it is, the more gentle the asimptotic progression is

	def __init__(self, model):
		self.model = model
		paramsList = []
		paramsList.append(np.random.uniform(-0.01,0.01,model.W_xh.shape))
		paramsList.append(np.random.uniform(-0.01,0.01,model.W_hh.shape))
		paramsList.append(np.random.uniform(-0.01,0.01,model.W_hy.shape))
		paramsList.append(np.random.uniform(-0.01,0.01,model.bh.shape))
		paramsList.append(np.random.uniform(-0.01,0.01,model.by.shape))
		self.bestPos = self.pos = np.concatenate(paramsList, axis=1)
		self.v = np.random.uniform(self.VMIN,self.VMAX,self.bestPos.shape)

	def update(self, gBest, gen):
		if gen < self.LINGEN:
			w = self.W0 - (self.W1/self.LINGEN) * gen
		else:
			w = self.W1 * math.exp((self.LINGEN - gen)/self.K)
		self.v = w * self.v + self.C1 * random.random() * (self.bestPos - self.pos) + self.C2 * random.random() * (gBest - self.pos)
		np.clip(self.v, self.VMIN, self.VMAX, out=self.v)
		self.pos += self.v
		np.clip(self.pos, -1, 1, out=self.pos) # we're not allowed outside the playground

	def copy(self):
		p = Particle(self.model)
		p.pos = self.pos.copy()
		p.bestPos = self.bestPos.copy()
		p.v = self.v.copy()
		return p


# Is fitness1 greater than fitness2?
def isGt(fitness1, fitness2):
	if fitness1[2] < fitness2[2]:
		return False
	elif fitness1[1] < fitness2[1]:
		return False
	else: # compare loss
		return fitness1[0] < fitness2[0]

class PSORNN:
	MAXGEN = 400 # max nr of generations of search

	def __init__(self, model, task):
		self.model = model
		self.task = task
		self.testX, self.testY = task.getData(10, 1000)

	def testModel(self):
		output = []
		for X in self.testX:
			self.model.reset(self.testX.shape[2]) # set to compute whole batch at once
			Y = self.model.forward(X)
			output.append(Y)
		output = np.array(output)
		loss = np.square(output - self.testY).sum() / self.testY.size
		allCorrect, bitsCorrect = self.task.analyzeRez(output, self.testY)
		return loss, allCorrect, bitsCorrect

	def fitness(self, particle):
		k = 0
		# for i, layer in enumerate(self.model.layers):
		self.model.W_xh = particle.pos[:,k:k+self.model.W_xh.shape[1]]
		k += self.model.W_xh.shape[1]
		self.model.W_hh = particle.pos[:,k:k+self.model.W_hh.shape[1]]
		k += self.model.W_xh.shape[1]
		self.model.W_hy = particle.pos[:,k:k+self.model.W_hy.shape[1]]
		k += self.model.W_hy.shape[1]
		self.model.bh = particle.pos[:,k:k+1]
		k += 1
		self.model.by = particle.pos[:,k:k+1]
		return self.testModel()

	def train(self, numPart):
		# Initialize particles
		P = []
		gBestFit = (9999,0,0)
		for i in range(0, numPart):
			p = Particle(self.model)
			P.append(p)
			p.fitn = self.fitness(p)
			if isGt(p.fitn, gBestFit):
				gBestFit = p.fitn
				bestP = p.copy() # so best particle remains unchanged
		epochUnchanged = 0
		print 'initialization done'
		gen = 0 # current generation
		while epochUnchanged < 10 and gen < self.MAXGEN:
			for p in P:
				p.update(bestP.bestPos, gen)
			unchanged = True
			for p in P:
				fitn = self.fitness(p)
				if isGt(fitn, p.fitn): # check if new position better
					p.bestPos = p.pos
					if isGt(fitn, gBestFit):
						print str(gBestFit)+" "+str(fitn)
						gBestFit = fitn
						bestP = p.copy()
						unchanged = False
						epochUnchanged = 0
				p.fitn = fitn
			if unchanged:
				print 'unchanged'
				epochUnchanged += 1
			gen += 1
		print self.fitness(bestP) # shortcut to place best parameters in model