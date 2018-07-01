
# coding: utf-8

# In[81]:


import numpy as np
np.random.seed(1234)
import time
import sys

# In[2]:


# from IPython.core.debugger import set_trace


# In[65]:

input_phrase = raw_input("Please enter a lowercased phrase of your choice: ")
target_phrase = np.array(list(input_phrase.lower()) + [" "])
target_len = len(target_phrase)


# In[66]:


import string


# In[67]:


vocab = np.array(list(set(string.ascii_lowercase).union(set(target_phrase))))


# In[68]:


vocab


# In[69]:


POP_SIZE = 500
MUTATION_RATE = 0.01

print "POPULATION SIZE:", POP_SIZE
print "MUTATION RATE:", MUTATION_RATE

# In[70]:


class Organism:
    def __init__(self, dna=None):
        if dna is None:
            self.dna = np.random.choice(vocab, target_len, replace=True)
        else:
            self.dna = dna
        self.name = "".join(self.dna)
        self.size = len(self.dna)
        self.mid = self.size // 2

    def fitness(self):
        return np.mean(self.dna == target_phrase)


# In[71]:


def create_population(n=POP_SIZE):
    return [Organism() for _ in range(n)]


# In[72]:


def create_pool(org_list):
    num = len(org_list)
    scores = np.array([o.fitness() for o in org_list])
    proba = scores / np.sum(scores)
    return np.random.choice(org_list, num, p=proba)


# In[73]:


def natural_selection(org_list):
    num = len(org_list)
    next_gen = []
    for _ in range(num):
        a, b = np.random.choice(org_list, 2, replace=False)
        child = crossover(a, b)
        child = mutate(child)
        next_gen.append(child)

    return next_gen


# In[74]:


def crossover(a, b):
    mid = a.mid
    dna = np.hstack((a.dna[:mid], b.dna[mid:]))
    return Organism(dna)


# In[75]:


def mutate(org):
    new_dna = []
    for gene in org.dna:
        if np.random.rand() < MUTATION_RATE:
            ## mutate
            new_dna.append(np.random.choice(vocab))
        else:
            new_dna.append(gene)
    return Organism(np.array(new_dna))


# In[76]:


def get_best(org_list):
    cur_best = -1
    for o in org_list:
        score = o.fitness()
        if score>cur_best:
            best = o
            cur_best = score

    return best


# In[79]:


def main():
    found = False
    generation = 0
    population = create_population()

    while not found:
        generation += 1
        best_org = get_best(population)
        if np.alltrue(best_org.dna == target_phrase):
            found = True
            break

        if generation%50==0:
            print "Generation: %d, Best organism: %s, Fitness: %f"%(generation, best_org.name, best_org.fitness())

        pool = create_pool(population)
        population = natural_selection(pool)

    print "FOUND"
    print "Generation: %d, Best organism: %s, Fitness: %f"%(generation, best_org.name, best_org.fitness())


# In[82]:


_st = time.time()
main()
print "Took", round((time.time()-_st)/60.0, 3), "mins."
sys.exit()

# # scratch

# In[17]:


population = create_population(25)


# In[18]:


for org in population:
    print org.name, org.fitness()


# In[19]:


pool = create_pool(population)
for org in pool:
    print org.name, org.fitness()


# In[20]:


child = crossover(pool[-1], pool[-2])


# In[21]:


child.name, child.fitness()


# In[22]:


best = get_best(population)
best.name, best.fitness()


# In[118]:


org = population[-2]


# In[119]:


new_org = mutate(org)


# In[120]:


new_org.name
