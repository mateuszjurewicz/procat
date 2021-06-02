"""
This module contains useful helper functions for running experiments
with set-to-sequence models.
"""
import numpy as np
import random
import string
import time
import torch
import copy
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from tqdm import tqdm


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {:,} trainable parameters'.format(params))


def get_permuted_batch(set_length, elem_dims, is_random=True):
    """
    Simple function for getting a 2-elem batch of a set and its random
    permutation.
    Can return the exact opposite order instead of a fully pseudo-random one.
    """
    in1 = np.random.rand(1, set_length, elem_dims)
    in2 = copy.deepcopy(in1)
    if is_random:
        for set in in2:
            np.random.shuffle(set)
    else:
        in2 = np.flip(in2, 1).astype(float)
    x1 = torch.from_numpy(in1).float()
    x2 = torch.from_numpy(in2).float()
    b = torch.stack([x1, x2])
    b = b.squeeze(1)
    return b


def get_example(a_dataloader, x_name='X', y_name='Y'):
    """
    Take a torch dataloader and get a single example.
    """
    a_batch = next(iter(a_dataloader))
    example_points = a_batch[x_name][0]
    example_solution = a_batch[y_name][0]

    return example_points, example_solution


def get_batch(a_dataloader, x_name='X', y_name='Y'):
    """
    Get a batch of points and solutions from a torch dataloader.
    """
    a_batch = next(iter(a_dataloader))
    batch_points = a_batch[x_name]
    batch_solutions = a_batch[y_name]

    return batch_points, batch_solutions


def train_epochs(a_model, a_model_optimizer, a_loss, a_dataloader, num_epochs,
                 allow_gpu=True, x_name='X', y_name='Y'):
    """
    Take a PtrNet model, an optimizer and a num_epochs and train it.
    """
    # cuda
    if torch.cuda.is_available() and allow_gpu:
        a_model.cuda()
        net = torch.nn.DataParallel(a_model,
                                    device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        print('CUDA available, using GPU:')
        print(torch.cuda.get_device_name(0))
    time.sleep(1)

    # epochs
    losses = []

    for epoch in range(num_epochs):
        batch_losses = []
        iterator = tqdm(a_dataloader, unit=' batches')

        for i_batch, sample_batched in enumerate(iterator):
            iterator.set_description(
                'Epoch {} / {}'.format(epoch + 1, num_epochs))

            train_batch = Variable(sample_batched[x_name])
            target_batch = Variable(sample_batched[y_name])

            if torch.cuda.is_available() and allow_gpu:
                train_batch = train_batch.cuda()
                target_batch = target_batch.cuda()

            o, p = a_model(train_batch)
            o = o.contiguous().view(-1, o.size()[-1])
            target_batch = target_batch.view(-1)

            loss = a_loss(o, target_batch)
            losses.append(loss.data)
            batch_losses.append(loss.data)

            a_model_optimizer.zero_grad()
            loss.backward()
            a_model_optimizer.step()

            # report
            iterator.set_postfix(avg_loss=' {:.5f}'.format(
                sum(batch_losses) / len(batch_losses)))

    # return model to cpu if needed
    if torch.cuda.is_available and allow_gpu:
        a_model.cpu()

    return sum(batch_losses) / len(batch_losses)
