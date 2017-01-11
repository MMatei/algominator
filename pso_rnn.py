import numpy as np
import random
import math
import time

class Particle:
	VMIN = -0.1
	VMAX = 0.1
	C1 = 2 # TODO: investigate C1, C2
	C2 = 2
	w = 0 # The inertial weight for adaptive PSO; computed once, globally
	W0 = 1.8 # W's are given by PSO-BP paper
	W1 = 0.6
	LINGEN = 100 # nr of generations in which w progresses linearly
	K = 100 # the bigger it is, the more gentle the asimptotic progression is

	def __init__(self, model):
		self.model = model
		self.pos = np.random.uniform(-1,1,#-0.01,0.01,
			(model.W_xh.shape[0],model.W_xh.shape[1]+model.W_hh.shape[1]+model.W_hy.shape[1]+2))
		self.bestPos = self.pos.copy()
		self.v = np.random.uniform(self.VMIN,self.VMAX,self.bestPos.shape)
		self.bestFitn = (9999,0,0)

	def updateW(self, gen):
		if gen < self.LINGEN:
			Particle.w = self.W0 - (self.W1/self.LINGEN) * gen
		else:
			Particle.w = self.W1 * math.exp((self.LINGEN - gen)/self.K)

	def updateFitness(self, fitness):
		self.fitn = fitness
		if isGt(fitness, self.bestFitn): # new best local position
			self.bestFitn = fitness
			self.bestPos = self.pos.copy()

	def update(self, gBest):
		self.v = Particle.w * self.v + self.C1 * random.random() * (self.bestPos - self.pos) + self.C2 * random.random() * (gBest - self.pos)
		np.clip(self.v, self.VMIN, self.VMAX, out=self.v)
		self.pos += self.v
		np.clip(self.pos, -1, 1, out=self.pos) # we're not allowed outside the playground

	def copy(self):
		p = Particle(self.model)
		p.pos = self.pos.copy()
		p.bestPos = self.bestPos.copy()
		p.v = self.v.copy()
		p.fitn = self.fitn
		p.bestFitn = self.bestFitn
		return p


class GParticle(Particle):
	def update(self, gBest, particles):
		force = 0
		for p in particles:
			force += random.random() * p.mass * self.mass * (p.pos - self.pos) #/ np.linalg.norm(p.pos*self.pos)
		# print force
		if force[0][0] == 0:
			print self.pos
			print self.mass
		acc = force / self.mass
		self.v = Particle.w * self.v + self.C1 * random.random() * acc + self.C2 * random.random() * (gBest - self.pos)
		np.clip(self.v, self.VMIN, self.VMAX, out=self.v)
		self.pos += self.v
		np.clip(self.pos, -1, 1, out=self.pos) # we're not allowed outside the playground

	# Can only be called after all particles have been updated with new fitness
	# Must be called before update
	def updateMass(self, bestFit, worstFit):
		# mass must be reduced to scalar; this is probably not the best idea
		self.mass = (self.fitn[0] - worstFit[0]) / (bestFit[0] - worstFit[0])
		if self.mass >= -1e-8 and self.mass < 1e-8:
			self.mass = 1e-8


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
		self.testX, self.testY = task.getData(10, 200)

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
		ts = time.time()
		for i in range(0, numPart):
			p = Particle(self.model)
			P.append(p)
			p.updateFitness(self.fitness(p))
			if isGt(p.fitn, gBestFit):
				gBestFit = p.fitn
				bestP = p.copy()
		epochUnchanged = 0
		print 'PSO initialization done in '+str(time.time()-ts)
		gen = 0 # current generation
		while epochUnchanged < 10 and gen < self.MAXGEN:
			P[0].updateW(gen) # update w for all particles - no sense repeating calc
			for p in P:
				p.update(bestP.bestPos)
			unchanged = True
			for p in P:
				fitn = self.fitness(p)
				p.updateFitness(fitn)
				if isGt(fitn, gBestFit):
					print str(gBestFit)+" "+str(fitn)
					gBestFit = fitn
					bestP = p.copy()
					unchanged = False
					epochUnchanged = 0
			if unchanged:
				print 'unchanged'
				epochUnchanged += 1
			gen += 1
		print self.fitness(bestP) # shortcut to place best parameters in model

class BPSORNN(PSORNN):
	def train(self, numPart):
		# Initialize particles
		P = []
		gBestFit = (9999,0,0)
		ts = time.time()
		for i in range(0, numPart):
			p = Particle(self.model)
			P.append(p)
			p.updateFitness(self.fitness(p))
			if isGt(p.fitn, gBestFit):
				gBestFit = p.fitn
				bestP = p.copy() # so best particle remains unchanged
		epochUnchanged = 0
		print 'PSO initialization done in '+str(time.time()-ts)
		gen = 0 # current generation
		while epochUnchanged < 10 and gen < self.MAXGEN:
			P[0].updateW(gen) # update w for all particles - no sense repeating calc
			for p in P:
				p.update(bestP.bestPos)
			unchanged = True
			for p in P:
				fitn = self.fitness(p)
				p.updateFitness(fitn)
				if isGt(fitn, gBestFit):
					print fitn
					gBestFit = fitn
					bestP = p.copy()
					unchanged = False
					epochUnchanged = 0
			if unchanged:
				print 'unchanged'
				epochUnchanged += 1
			gen += 1
		print self.fitness(bestP) # shortcut to place best parameters in model
		# Doing backpropagation to inch towards better results
		# TODO: investigate number of epochs; will outshine other methods since BP is so damn good
		trainX, trainY = self.task.getData(10, 2100)
		batchSz = 30 # TODO: this is currently determined by model
		for i in range(0,70):
			self.model.reset()
			seqX = trainX[:,:,batchSz*i:batchSz*(i+1)]
			Y = []
			for X in seqX:
				Y.append(self.model.forward(X))
			dY = np.array(Y) - trainY[:,:,batchSz*i:batchSz*(i+1)]
			self.model.backward(seqX, dY)
			self.model.updateParams(0.1)

class GPSORNN(PSORNN):

	def train(self, numPart):
		# Initialize particles
		P = []
		gBestFit = (9999,0,0)
		gWorstFit = (0, 100, 100)
		ts = time.time()
		for i in range(0, numPart):
			p = GParticle(self.model)
			P.append(p)
			p.updateFitness(self.fitness(p))
			if isGt(p.fitn, gBestFit):
				gBestFit = p.fitn
				bestP = p.copy() # so best particle remains unchanged
			if isGt(gWorstFit, p.fitn):
				gWorstFit = p.fitn
		epochUnchanged = 0
		print 'PSO initialization done in '+str(time.time()-ts)
		gen = 0 # current generation
		while epochUnchanged < 10 and gen < self.MAXGEN:
			for p in P:
				p.updateMass(gBestFit, gWorstFit)
			P[0].updateW(gen) # update w for all particles - no sense repeating calc
			for p in P:
				p.update(bestP.bestPos, P)
			unchanged = True
			for p in P:
				fitn = self.fitness(p)
				p.updateFitness(fitn)
				if isGt(fitn, gBestFit):
					gBestFit = fitn
					print gBestFit
					bestP = p.copy()
					unchanged = False
					epochUnchanged = 0
				if isGt(gWorstFit, fitn):
					gWorstFit = fitn
			if unchanged:
				print 'unchanged'
				epochUnchanged += 1
			gen += 1
		print self.fitness(bestP) # shortcut to place best parameters in model
