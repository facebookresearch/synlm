# Taken from https://github.com/ryan112358/private-pgm/blob/master/mechanisms/aim.py on 2023/03/29
# class AIM is modified to return model instead of synthetic data
# also added compute_cross_entropy

import numpy as np
import itertools
from mbi import GraphicalModel, FactoredInference
from pgm.mechanism import Mechanism
from hdmm.matrix import Identity
from src.utils import timeit

def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(1,len(s)+1))

def downward_closure(Ws):
    ans = set()
    for proj in Ws:
        ans.update(powerset(proj))
    return list(sorted(ans, key=len))

def hypothetical_model_size(domain, cliques):
    model = GraphicalModel(domain, cliques)
    return model.size * 8 / 2**20


def compile_workload(workload):
    def score(cl):
        return sum(len(set(cl)&set(ax)) for ax in workload)
    return { cl : score(cl) for cl in downward_closure(workload) }

def filter_candidates(candidates, model, size_limit):
    ans = { }
    free_cliques = downward_closure(model.cliques)
    for cl in candidates:
        cond1 = hypothetical_model_size(model.domain, model.cliques + [cl]) <= size_limit
        cond2 = cl in free_cliques
        if cond1 or cond2:
            ans[cl] = candidates[cl]
    return ans


@timeit
def compute_cross_entropy(model, data):
    logZ = model.belief_propagation(model.potentials, logZ=True)
    log_probas = np.zeros(data.records)

    for cl in model.cliques:
        P = model.potentials[cl].values
        idxs = data.project(cl).df.values.astype(int)
        log_probas += np.array([P[tuple(i)] for i in idxs])

    return logZ - log_probas

class AIM(Mechanism):
    def __init__(self,epsilon,delta,prng=None,rounds=None,max_model_size=80,structural_zeros={}):
        super(AIM, self).__init__(epsilon, delta, prng)
        self.rounds = rounds
        self.max_model_size = max_model_size
        self.structural_zeros = structural_zeros

    def worst_approximated(self, candidates, answers, model, eps, sigma):
        errors = {}
        sensitivity = {}
        for cl in candidates:
            wgt = candidates[cl]
            x = answers[cl]
            bias = np.sqrt(2/np.pi)*sigma*model.domain.size(cl)
            xest = model.project(cl).datavector()
            errors[cl] = wgt * (np.linalg.norm(x - xest, 1) - bias)
            sensitivity[cl] = abs(wgt) 

        max_sensitivity = max(sensitivity.values()) # if all weights are 0, could be a problem
        return self.exponential_mechanism(errors, eps, max_sensitivity)

    def run(self, data, W):
        rounds = self.rounds or 16*len(data.domain)
        workload = [cl for cl, _ in W]
        candidates = compile_workload(workload)
        answers = { cl : data.project(cl).datavector() for cl in candidates }

        oneway = [cl for cl in candidates if len(cl) == 1]

        sigma = np.sqrt(rounds / (2*0.9*self.rho))
        epsilon = np.sqrt(8*0.1*self.rho/rounds)
       
        measurements = []
        print('Initial Sigma', sigma)
        rho_used = len(oneway)*0.5/sigma**2
        for cl in oneway:
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma,x.size)
            I = Identity(y.size) 
            measurements.append((I, y, sigma, cl))

        zeros = self.structural_zeros
        engine = FactoredInference(data.domain,iters=1000,warm_start=True,structural_zeros=zeros)
        engine.estimate = timeit(engine.estimate)
        model = engine.estimate(measurements)

        t = 0
        terminate = False
        while not terminate:
            t += 1
            if self.rho - rho_used < 2*(0.5/sigma**2 + 1.0/8 * epsilon**2):
                # Just use up whatever remaining budget there is for one last round
                remaining = self.rho - rho_used
                sigma = np.sqrt(1 / (2*0.9*remaining))
                epsilon = np.sqrt(8*0.1*remaining)
                terminate = True

            rho_used += 1.0/8 * epsilon**2 + 0.5/sigma**2
            size_limit = self.max_model_size*rho_used/self.rho

            small_candidates = filter_candidates(candidates, model, size_limit)
            cl = self.worst_approximated(small_candidates, answers, model, epsilon, sigma)

            n = data.domain.size(cl)
            Q = Identity(n) 
            x = data.project(cl).datavector()
            y = x + self.gaussian_noise(sigma, n)
            measurements.append((Q, y, sigma, cl))
            z = model.project(cl).datavector()

            model = engine.estimate(measurements)
            print("Train XE", compute_cross_entropy(model, data).mean())
            w = model.project(cl).datavector()
            print('Selected',cl,'Size',n,'Budget Used',rho_used/self.rho)
            if np.linalg.norm(w-z, 1) <= sigma*np.sqrt(2/np.pi)*n:
                print('(!!!!!!!!!!!!!!!!!!!!!!) Reducing sigma', sigma/2)
                sigma /= 2
                epsilon *= 2

        print('Fitting model...')
        engine.iters = 2500
        model = engine.estimate(measurements)

        return model