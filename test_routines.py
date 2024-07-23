from scipy import stats
import numpy as np

from scores_from_ranking import ScoresFromRanking

def create_data(n_items, n_rankings, max_items_per_ranking=None):
    # create a random ranked list of items.
    true_rankings = np.arange(n_items)
    np.random.shuffle(true_rankings)

    if max_items_per_ranking is None:
        max_items_per_ranking = 2

    # pick a random subsequence of `true_rankings`
    rankings = []

    new_rankings = np.arange(n_items)
    
    for i in range(n_rankings):
        # create a newly shuffled list of items
        np.random.shuffle(new_rankings)
        rankings.append(true_rankings[np.sort(new_rankings[:np.random.randint(2, 1+max_items_per_ranking)])])
        
    return true_rankings, rankings

def compute_correlations(n_rankings, n_items, n_random_trials):
    correlations = []
    for _ in range(n_random_trials):
        true_rankings, rankings = create_data(n_items, n_rankings)
        sfr = ScoresFromRanking(n_items, rankings)
        sfr.optimize()
        corr = stats.pearsonr(sfr.get_ranking(), true_rankings).statistic
        correlations.append(corr)
    return [n_rankings, np.mean(correlations), np.std(correlations)]