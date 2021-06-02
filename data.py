"""
This module contains the data generators and data loaders for the
tasks planned for the comparative analysis publication.
"""
import itertools
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from itertools import chain
from scipy.spatial import distance
from scipy.stats import spearmanr, kendalltau
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


def get_example(a_dataloader):
    """
    Take a torch dataloader and get a single example.
    """
    a_batch = next(iter(a_dataloader))
    example_points = a_batch['X'][0]
    example_solution = a_batch['Y'][0]

    return example_points, example_solution


def get_batch(a_dataloader):
    """
    Get a batch of points and solutions from a torch dataloader.
    """
    a_batch = next(iter(a_dataloader))
    batch_points = a_batch['X']
    batch_solutions = a_batch['Y']

    return batch_points, batch_solutions


def train_toy_epochs(a_model, a_model_optimizer, a_loss,
                     a_dataloader, num_epochs, allow_gpu=False):
    """
    Train a toy model.
    """
    # epochs
    losses = []

    for epoch in range(num_epochs):
        batch_losses = []
        iterator = tqdm(a_dataloader, unit=' batches')

        for i_batch, sample_batched in enumerate(iterator):
            iterator.set_description(
                'Epoch {} / {}'.format(epoch + 1, num_epochs))

            train_batch = Variable(sample_batched['X'])
            target_batch = Variable(sample_batched['Y'])

            if torch.cuda.is_available() and allow_gpu:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()

            pred = a_model(train_batch)
            pred = pred.float()

            target_batch = target_batch.view(-1).float()

            loss = a_loss(pred, target_batch)
            losses.append(loss.data)
            batch_losses.append(loss.data)

            a_model_optimizer.zero_grad()
            loss.backward()
            a_model_optimizer.step()

            # report
            iterator.set_postfix(avg_loss=' {:.5f}'.format(
                sum(batch_losses) / len(batch_losses)))


def test_model_custom(a_model, a_dataloader,
                      comparison_func=None, print_every=500,
                      x_name='X', y_name='Y'):
    """
    I would now like a function that takes a dataloader of test data and a model package,
    making it predict on each and then outputs the average loss.
    """
    # metrics, placeholders
    num_examples = len(a_dataloader.dataset)
    individual_scores = []
    counter = 0

    # iterate over dataset to predict and track
    # no grad here
    with torch.no_grad():
        for single_batch in a_dataloader:

            train_batch = Variable(single_batch[x_name])
            target_batch = Variable(single_batch[y_name])

            # predict
            o, batched_predictions = a_model(train_batch)

            # track
            for idx, model_solution in enumerate(batched_predictions):

                # compare solutions (might need a custom function per dataset)
                current_score = comparison_func(prediction=model_solution,
                                                solution=target_batch[idx])
                individual_scores.append(current_score)

                # update counter & report
                counter += 1
                if counter % print_every == 0:
                    print('... Calculating example {} / {} ...'.format(
                        counter, num_examples
                    ))

        # TODO: post-process (e.g. aggregate)
        final_score = sum(individual_scores) / num_examples

    return final_score, individual_scores


def compare_solved_sort_unique(prediction, solution):
    """
    Take a single model prediction and a single actual soltuion,
    compare them in a way that return a single number marking the quality
    of the prediction.
    """
    correct_preds = 0
    total_preds = len(prediction)
    for i, p in enumerate(prediction):
        if p == solution[i]:
            correct_preds += 1

    return correct_preds / total_preds


class FloatSortDataset(Dataset):
    """
    Random (seedable) integer sorting dataset, with length specified durining initialization.
    We only generate floats between 0 and 1. we don't care about actual 0 ocurring (should be rare)
    nor about duplicates (equally rare).
    """

    def __init__(self, data_size, seq_len, solve=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = self.sort_solver
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.from_numpy(self.data['Y'][idx]).long()

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of float_list and their vector solutions
        """
        floats_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                'Data points %i/%i' % (i + 1, self.data_size))
            floats_list.append(np.random.random(self.seq_len))

        solutions_iter = tqdm(floats_list, unit='solve')
        if self.solve:
            # get solutions
            for i, floats in enumerate(solutions_iter):
                solutions_iter.set_description(
                    'Solved %i/%i' % (i + 1, len(floats_list)))
                solution = self.solver(floats)
                solutions.append(solution)
        else:
            solutions = None

        return {'X': floats_list, 'Y': solutions}

    @staticmethod
    def sort_solver(floats):
        """
        Take a list of floats and sort them.
        Return the sorted indices as np array.
        """
        return np.argsort(floats)


class EuclideanTSPDataset(Dataset):
    """
    Random TSP dataset, in 2 dimensions.
    """

    def __init__(self, data_size, seq_len, solve=True):
        self.data_size = data_size
        self.seq_len = seq_len
        self.solve = solve
        self.solver = self.held_karp_solver
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.from_numpy(
            self.data['Y'][idx]).long() if self.solve else None

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        points_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                'Data points %i/%i' % (i + 1, self.data_size))
            points_list.append(np.random.random((self.seq_len, 2)))
        solutions_iter = tqdm(points_list, unit='solve')
        if self.solve:
            for i, points in enumerate(solutions_iter):
                solutions_iter.set_description(
                    'Solved %i/%i' % (i + 1, len(points_list)))
                solutions.append(self.solver(points))
        else:
            solutions = None

        return {'X': points_list, 'Y': solutions}

    def _to1hotvec(self, points):
        """
        :param points: List of integers representing the points indexes
        :return: Matrix of One-Hot vectors
        """
        vec = np.zeros((len(points), self.seq_len))
        for i, v in enumerate(vec):
            v[points[i]] = 1

        return vec

    @staticmethod
    def mlalevic__solver(points):
        """
        Dynamic programing solution for TSP - O(2^n*n^2)
        https://gist.github.com/mlalevic/6222750
        :param points: List of (x, y) points
        :return: Optimal solution
        """

        def length(x_coord, y_coord):
            return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

        # Calculate all lengths
        all_distances = [[length(x, y) for y in points] for x in points]
        A = {(frozenset([0, idx + 1]), idx + 1): (dist, [0, idx + 1]) for
             idx, dist in enumerate(all_distances[0][1:])}
        cnt = len(points)
        for m in range(2, cnt):
            B = {}
            for S in [frozenset(C) | {0} for C in
                      itertools.combinations(range(1, cnt), m)]:
                for j in S - {0}:
                    B[(S, j)] = min([(A[(S - {j}, k)][0] + all_distances[k][j],
                                      A[(S - {j}, k)][1] + [j])
                                     for k in S if k != 0 and k != j])
            A = B
        res = min(
            [(A[d][0] + all_distances[0][d[1]], A[d][1]) for d in iter(A)])
        return np.asarray(res[1])

    @staticmethod
    def held_karp_solver(points):
        """
        Implementation of Held-Karp, an algorithm that solves the Traveling
        Salesman Problem using dynamic programming with memoization.
        Parameters:
            points: points towards distance matrix
        Returns:
            A path.
        """
        def length(x_coord, y_coord):
            return np.linalg.norm(np.asarray(x_coord) - np.asarray(y_coord))

        # Calculate all lengths
        dists = [[length(x, y) for y in points] for x in points]
        n = len(dists)

        # Maps each subset of the nodes to the cost to reach that subset, as well
        # as what node it passed before reaching this subset.
        # Node subsets are represented as set bits.
        C = {}

        # Set transition cost from initial state
        for k in range(1, n):
            C[(1 << k, k)] = (dists[0][k], 0)

        # Iterate subsets of increasing length and store intermediate results
        # in classic dynamic programming manner
        for subset_size in range(2, n):
            for subset in itertools.combinations(range(1, n), subset_size):
                # Set bits for all nodes in this subset
                bits = 0
                for bit in subset:
                    bits |= 1 << bit

                # Find the lowest cost to get to this subset
                for k in subset:
                    prev = bits & ~(1 << k)

                    res = []
                    for m in subset:
                        if m == 0 or m == k:
                            continue
                        res.append((C[(prev, m)][0] + dists[m][k], m))
                    C[(bits, k)] = min(res)

        # We're interested in all bits but the least significant (the start state)
        bits = (2 ** n - 1) - 1

        # Calculate optimal cost
        res = []
        for k in range(1, n):
            res.append((C[(bits, k)][0] + dists[k][0], k))
        opt, parent = min(res)

        # Backtrack to find full path
        path = []
        for i in range(n - 1):
            path.append(parent)
            new_bits = bits & ~(1 << parent)
            _, parent = C[(bits, parent)]
            bits = new_bits

        # Add implicit start state
        path.append(0)

        return np.asarray(list(reversed(path)))


def get_tour_length(points, solution):
    """
    Calculate the length of the tour between points,
    given their order provided in the solution.
    """
    tour_length = 0

    # convert to numpy
    points = points.numpy()
    solution = solution.numpy()

    # order them according to solutions
    # (using numpy ordering)
    ordered_points = points[solution]

    # calculate addiitive distance
    for i in range(len(ordered_points)):

        # end condition
        if i + 1 == len(ordered_points):
            break
        else:
            tour_length += distance.euclidean(ordered_points[i],
                                              ordered_points[i + 1])

    return tour_length


class RangeIntegersDataset(Dataset):
    """
    From Janossy 2019:
    "the range task also receives a sequence of n integers
    drawn uniformly at random with replacement from range <min, max>,
    and tries to predict the range (the difference between the
    maximum and minimum)."
    """

    def __init__(self, data_size, seq_len=5, min_incl=0, max_incl=99):
        self.data_size = data_size
        self.seq_len = seq_len
        self.min_incl = min_incl
        self.max_incl = max_incl
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.tensor([
            self.data['Y'][idx]])

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        integers_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                'Data points %i/%i' % (i + 1, self.data_size))
            integers_list.append(np.random.randint(self.min_incl,
                                                   self.max_incl + 1,
                                                   self.seq_len))
        solutions_iter = tqdm(integers_list, unit='solve')

        # solve
        for i, integers in enumerate(solutions_iter):
            solutions_iter.set_description(
                'Solved %i/%i' % (i + 1, len(integers_list)))
            solutions.append(np.ptp(integers))

        return {'X': integers_list, 'Y': solutions}


class UniqueSumIntegersDataset(Dataset):
    """
    From Janossy 2019:
    "the unique sum task receives a sequence of 10 integers, sampled uniformly
     with replacement from {0, 1, . . . , 9}, and predicts the sum of all
    unique elements"
    """

    def __init__(self, data_size, seq_len=10, min_incl=0, max_incl=9):
        self.data_size = data_size
        self.seq_len = seq_len
        self.min_incl = min_incl
        self.max_incl = max_incl
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.tensor([
            self.data['Y'][idx]])

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        integers_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                'Data points %i/%i' % (i + 1, self.data_size))
            integers_list.append(np.random.randint(self.min_incl,
                                                   self.max_incl + 1,
                                                   self.seq_len))
        solutions_iter = tqdm(integers_list, unit='solve')

        # solve
        for i, integers in enumerate(solutions_iter):
            solutions_iter.set_description(
                'Solved %i/%i' % (i + 1, len(integers_list)))
            solutions.append(np.sum(np.unique(integers)))

        return {'X': integers_list, 'Y': solutions}


class UniqueCountIntegersDataset(Dataset):
    """
    From Janossy 2019:
    "the unique count task also receives a sequence of repeating elements from
    {0, 1, . . . , 9}, distributed in the same was as with the unique sum task,
    and predicts the number of unique elements"
    """

    def __init__(self, data_size, seq_len=10, min_incl=0, max_incl=9):
        self.data_size = data_size
        self.seq_len = seq_len
        self.min_incl = min_incl
        self.max_incl = max_incl
        self.data = self._generate_data()

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.tensor([
            self.data['Y'][idx]])

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self):
        """
        :return: Set of points_list ans their One-Hot vector solutions
        """
        integers_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                'Data points %i/%i' % (i + 1, self.data_size))
            integers_list.append(np.random.randint(self.min_incl,
                                                   self.max_incl + 1,
                                                   self.seq_len))
        solutions_iter = tqdm(integers_list, unit='solve')

        # solve
        for i, integers in enumerate(solutions_iter):
            solutions_iter.set_description(
                'Solved %i/%i' % (i + 1, len(integers_list)))
            solutions.append(len(np.unique(integers)))

        return {'X': integers_list, 'Y': solutions}


class SpecialFloatSortUniqueDoublePrime(Dataset):
    """
    Generate a dataset of sets of floating point numbers (only .0, .25, .5 and .75),
    where the target is the sorted sequence of only unique elements of this set,
    with prime numbers being repeated twice in the output sequence.
    """

    def __init__(self, data_size, seq_len, min_val, max_val, rounder_multiple,
                 start_value, end_value, pad_value, max_length, solve=True):
        self.data_size = data_size
        self.min_val = min_val
        self.max_val = max_val
        self.seq_len = seq_len
        self.solve = solve
        self.solver = self.generate_y_from_x_with_specials
        self.rounder = self.round_nearest
        self.data = self._generate_data(min_val, max_val, seq_len,
                                        rounder_multiple, start_value,
                                        end_value, pad_value, max_length,
                                        self.round_nearest)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        tensor = torch.from_numpy(self.data['X'][idx]).float()
        solution = torch.from_numpy(self.data['Y'][idx]).long()

        sample = {'X': tensor, 'Y': solution}

        return sample

    def _generate_data(self, minval, maxval, length, rounder_multiple, start,
                       end, padder, max_length, rounding_func, debug=None):
        """
        :return: Set of float_list and their vector solutions
        """
        floats_list = []
        solutions = []
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description(
                'Data points %i/%i' % (i + 1, self.data_size))
            floats_list.append(
                self.generate_x_with_specials(minval, maxval, length,
                                              rounder_multiple, start, end,
                                              padder, rounding_func)
            )

        solutions_iter = tqdm(floats_list, unit='solve')
        if self.solve:
            # get solutions
            for i, floats in enumerate(solutions_iter):
                solutions_iter.set_description(
                    'Solved %i/%i' % (i + 1, len(floats_list)))
                solution = self.solver(floats, max_length)
                solutions.append(solution)
        else:
            solutions = None

        return {'X': floats_list, 'Y': solutions}

    @staticmethod
    def round_nearest(x, a):
        # decimal multiple defined here
        return round(x / a) * a

    @staticmethod
    def generate_x_with_specials(minval, maxval, length, rounder_multiple,
                                 start, end, padder, rounding_f, debug=None):

        # generate random
        x = [rounding_f(e, rounder_multiple) for e in
             np.random.uniform(low=minval, high=maxval, size=(length,))]

        # can override when debugging
        if debug:
            x = debug

        # insert special SOS, EOS and PAD.
        x.insert(0, padder)
        x.insert(0, end)
        x.insert(0, start)

        # to numpy
        x = np.asarray(x)

        return x

    def generate_y_from_x_with_specials(self, an_x, max_length):
        # ignore 3 special tokens
        t = set(an_x[3:])
        t = sorted(t)
        t = list(chain.from_iterable(
            [e] * 2 if self.is_prime(e) else [e] for e in t))
        y = []
        for e in t:
            y.append(list(an_x).index(e))

        # add special tokens back and pad
        y.insert(0, 0)  # start token always zeroth
        y.append(1)  # ender always 1st elem in x

        padding_needed = max_length - len(y)
        for i in range(padding_needed):
            y.append(2)  # padder always second elem in x

        return np.asarray(y)

    @staticmethod
    def is_prime(n):
        if round(n) != n: return False  # floats can't be prime
        if n == 2 or n == 3: return True
        if n % 2 == 0 or n < 2: return False
        for i in range(3, int(n ** 0.5) + 1, 2):  # only odd numbers
            if n % i == 0:
                return False

        return True

    def check_dupli_prime(self, x):
        """
        Check for duplicates and primes.
        """
        x = x.tolist()
        if len(x) != len(set(x)):
            print('Duplicate(s) found')
            counts = {float(k): 0 for k in x}
            for e in x:
                counts[float(e)] += 1
            print(sorted([(k, v) for k, v in counts.items() if v > 1]))
        else:
            print('No duplication found')
        for e in x:
            ef = float(e)
            if self.is_prime(ef):
                print('Prime found: ', ef)

    @staticmethod
    def restore_from_xy(an_x, a_y):
        an_x = list(an_x)
        r = []
        for i in a_y:
            r.append(float(an_x[i]))
        return r


def get_single_kendall_tau(single_y, single_pred):
    """Returns the Kendall Tau score for a single prediction"""
    def get_x_idx_and_rank_dict(a_y):
        x_idx_and_rank = dict()
        for i, e in enumerate(a_y):
            x_idx_and_rank[int(e)] = int(i)
        return x_idx_and_rank

    # map ranks to idx in x
    y = get_x_idx_and_rank_dict(single_y)
    p = get_x_idx_and_rank_dict(single_pred)

    # sort
    ranks_y = [y[key] for key in sorted(y.keys(), reverse=True)]
    ranks_p = [p[key] for key in sorted(p.keys(), reverse=True)]

    # compare
    tau, _ = kendalltau(ranks_y, ranks_p)

    return tau


def get_single_spearman_rho(single_y, single_pred):
    """Returns the Spearman Rho score for a single prediction"""
    def get_x_idx_and_rank_dict(a_y):
        x_idx_and_rank = dict()
        for i, e in enumerate(a_y):
            x_idx_and_rank[int(e)] = int(i)
        return x_idx_and_rank

    # map ranks to idx in x
    y = get_x_idx_and_rank_dict(single_y)
    p = get_x_idx_and_rank_dict(single_pred)

    # sort
    ranks_y = [y[key] for key in sorted(y.keys(), reverse=True)]
    ranks_p = [p[key] for key in sorted(p.keys(), reverse=True)]

    # compare
    tau, _ = spearmanr(ranks_y, ranks_p)

    return tau


def get_batch_rank_correlation(y_dataloader, a_model, rank_correlation_func,
                               print_every=1000):
    """
    Take a dataloader and a model, predict, return average of chosen rank
    correlation metric.
    """
    ranks = []
    c = 0

    # go through every batch
    for batch_x, batch_y in y_dataloader:

        # predict
        _, batch_pred = a_model(batch_x)

        # get individual correlation ranks
        for i, y in enumerate(batch_y):
            c += 1
            pred = batch_pred[i]
            rank = rank_correlation_func(y, pred)
            ranks.append(rank)

            if c % print_every == 0:
                print("{} / {}".format(c, len(y_dataloader.dataset)))

    # aggregate (avg)
    r = sum(ranks) / len(ranks)
    return r


def check_is_prediction_invalid(single_pred_as_batch):
    """
    Take tensors of shape [n], return bool if no repetition of elements (as if it was masked).
    """
    is_repeated = True

    n_target = len(single_pred_as_batch)
    n_ranked = len(single_pred_as_batch.unique())

    if n_target == n_ranked:
        is_repeated = False

    return is_repeated


# first, we need a function for predicting on an entire dataloader
def get_batch_rank_correlation_and_perc_valid(y_dataloader, a_model,
                                              rank_correlation_func,
                                              print_every=1000):
    """
    Take a dataloader and a model, predict, return average of chosen rank
    correlation metric and what % of predictions could even be tested.
    """
    ranks = []
    valid_predictions = 0
    c = 0

    # go through every batch
    for batch_x, batch_y in y_dataloader:

        # predict
        _, batch_pred = a_model(batch_x)

        # get individual correlation ranks
        for i, y in enumerate(batch_y):
            c += 1
            pred = batch_pred[i]

            # figure out if prediction has no repetition (predicts a rank for each element)
            prediction_is_repeated = check_is_prediction_invalid(pred)

            if not prediction_is_repeated:
                valid_predictions += 1
                rank = rank_correlation_func(y, pred)
                ranks.append(rank)

            if c % print_every == 0:
                print("{} / {}".format(c, len(y_dataloader.dataset)))

    # aggregate (avg)
    if len(ranks) > 0:
        r = sum(ranks) / len(ranks)
    else:
        r = 0

    # % of valid
    perc_valid = round(valid_predictions * 100 / c, 2)
    return r, perc_valid