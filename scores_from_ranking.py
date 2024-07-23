#
# we define a class `ScoresFromRanking` that takes rankings of subsets of items and returns a score for each item.
# the score of each item must satisfy the ranking constraints for all subsets in which it appears.
#

import numpy as np
import torch
from torch import nn

import copy

class ScoresFromRanking(nn.Module):
    def __init__(self, 
                 n_items: int,
                 rankings: list,
                 margin=1e-5):
        super().__init__()
        self.n_items = n_items
        self.rankings = rankings
        self.margin = margin

        self.scores = torch.randn(n_items, requires_grad=True)

    def constraint_score(self, ranking):
        cs = 0.

        # create all pairs of items in the ranking
        for i in range(len(ranking)):
            for j in range(i+1, len(ranking)):
                cs += (-self.scores[ranking[i]] + (self.scores[ranking[j]]+self.margin)).clamp(min=0)

        return cs
    
    def overall_score(self):
        return sum([self.constraint_score(r) for r in self.rankings])
    
    def forward(self):
        return self.overall_score()
    
    def optimize(self, n_iter=100, lr=0.1):
        optimizer = torch.optim.Adam([self.scores], lr=lr)
        
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = self()
            loss.backward()
            optimizer.step()

    def get_scores(self):
        return self.scores.detach().numpy()
    
    def get_ranking(self):
        return np.argsort(-self.get_scores())
    
# example usage
n_items = 10
# create a random ranked list of items.
true_rankings = np.arange(n_items)
np.random.shuffle(true_rankings)
# pick a random subsequence of `true_rankings`
rankings = []

for i in range(10):
    rankings.append([true_rankings[j] 
                     for j 
                     in np.sort(np.unique(np.random.randint(0, 
                                                  n_items, 
                                                  np.random.randint(2, n_items // 2))))])
    
sfr = ScoresFromRanking(n_items, rankings)
initial_rankings = copy.deepcopy(sfr.get_ranking())
sfr.optimize()

print(f'True ranking: {true_rankings}')

# print all rankings
for i, r in enumerate(rankings):
    print(f'Sampled ranked subsets {i}: {r}')

print(f'Initial ranking: {initial_rankings}')
print(f'Estimated ranking: {sfr.get_ranking()}')


