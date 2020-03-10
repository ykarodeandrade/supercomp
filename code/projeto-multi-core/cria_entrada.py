import random
import numpy as np

n_alunos = 45
n_proj = 15
n_escolhas = 3

print(n_alunos, n_proj, n_escolhas)

for i in range(n_alunos):
    choices = np.random.permutation(n_proj)
    print(' '.join([str(choices[c]) for c in range(n_escolhas) ]))
