import time
from pso_rnn import PSORNN, Particle, isGt

class ChaosPSORNN(PSORNN):
	def chaosLocalSearch(self, p):
		x = p.pos
		for i in range(0,100):
			# chaos search takes place in a normalized space [0,1]
			cx = (p.pos + 1) / 2
			cx = 4 * cx * (1 - cx)
			p.pos = -1 + cx * 2
			f = self.fitness(p)
			if isGt(f, p.fitn):
				print 'Chaos improvement ' + str(f)
				p.updateFitness(f)
				return p
		p.pos = x
		return p

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
					print fitn
					bestP = self.chaosLocalSearch(p.copy())
					gBestFit = bestP.fitn
					unchanged = False
					epochUnchanged = 0
			if unchanged:
				print str(gen)+') unchanged'
				epochUnchanged += 1
			gen += 1
		print self.fitness(bestP) # shortcut to place best parameters in model