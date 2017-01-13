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

	def __init__(self, posShape):
		self.posShape = posShape
		self.pos = np.random.uniform(-1,1,#-0.01,0.01,
			posShape)
		self.bestPos = self.pos.copy()
		self.v = np.random.uniform(self.VMIN, self.VMAX, self.pos.shape)
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
		p = Particle(self.posShape)
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
	MAXGEN = 300 # max nr of generations of search

	def __init__(self, model, task):
		self.model = model
		self.task = task
		self.testX, self.testY = task.getData(10, 200)
		self.posShape = 0
		self.shapes = [] # so there's no need to look them up again
		for layer in model.layers:
			self.posShape += (layer.W_xh.shape[1]+layer.W_hh.shape[1]+layer.W_hy.shape[1]+2) * layer.W_xh.shape[0]
			self.shapes.append(layer.W_xh.shape)
			self.shapes.append(layer.W_hh.shape)
			self.shapes.append(layer.W_hy.shape)
			self.shapes.append(layer.bh.shape)
			self.shapes.append(layer.by.shape)

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
		i = 0
		k = 0
		shapes = self.shapes
		for layer in self.model.layers:
			s = shapes[i][0]*shapes[i][1]
			layer.W_xh = np.reshape(particle.pos[k:k+s], shapes[i])
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			layer.W_hh = np.reshape(particle.pos[k:k+s], shapes[i])
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			layer.W_hy = np.reshape(particle.pos[k:k+s], shapes[i])
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			layer.bh = np.reshape(particle.pos[k:k+s], shapes[i])
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			layer.by = np.reshape(particle.pos[k:k+s], shapes[i])
			k += s
			i += 1
		return self.testModel()

	def train(self, numPart):
		# Initialize particles
		P = []
		gBestFit = (9999,0,0)
		ts = time.time()
		for i in range(0, numPart):
			p = Particle(self.posShape)
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
					print str(gen)+') '+str(fitn)
					gBestFit = fitn
					bestP = p.copy()
					unchanged = False
					epochUnchanged = 0
			if unchanged:
				print str(gen)+') unchanged'
				epochUnchanged += 1
			gen += 1
		print self.fitness(bestP) # shortcut to place best parameters in model

class BPSORNN(PSORNN):
	def modelToPart(self, p):
		i = 0
		k = 0
		shapes = self.shapes
		for layer in self.model.layers:
			s = shapes[i][0]*shapes[i][1]
			p.pos[k:k+s] = layer.W_xh.flatten()
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			p.pos[k:k+s] = layer.W_hh.flatten()
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			p.pos[k:k+s] = layer.W_hy.flatten()
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			p.pos[k:k+s] = layer.bh.flatten()
			k += s
			i += 1
			s = shapes[i][0]*shapes[i][1]
			p.pos[k:k+s] = layer.by.flatten()
			k += s
			i += 1
		p.bestPos = p.pos.copy()

	def train(self, numPart):
		# Initialize particles
		P = []
		gBestFit = (9999,0,0)
		ts = time.time()
		for i in range(0, numPart):
			p = Particle(self.posShape)
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
					# Use backprop to train bestP a bit
					self.model.reset(self.testX.shape[2])
					Y = []
					for X in self.testX:
						Y.append(self.model.forward(X))
					dY = np.array(Y) - self.testY
					self.model.backward(dY, 0.1)
					fitn = self.testModel()
					print str(fitn) + ' ' + str(gBestFit)
					if isGt(fitn, gBestFit): # if improved fitness, change bestP accordingly
						print 'BP increase ' + str(fitn)
						gBestFit = fitn
						self.modelToPart(bestP)
			if unchanged:
				print 'unchanged'
				epochUnchanged += 1
			gen += 1
		print self.fitness(bestP) # shortcut to place best parameters in model

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
