"""
This module contains modularized model architectures for the comparative
analysis publications. The idea is to store all implementations in one place,
solely for the purpose of comparative set-to-sequence experiments,
without unintentionally making breaking changes to other notebooks.

Additionally, these implementations should be clean enough for easy
readability in a github repository made available as part of the publication.

Finally, depending on the data task, models might require slight adjustments,
which I hope to make available via parameters, so that we don't have to have
a model per task.

For perfect reproducibility, I will probably also have jupyter notebooks with
the entire needed model code as backup.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter


###############################################################################
###############################################################################
# 0. NAIVE SORTER
# The simplest way to ensure permutation invariance is to sort all elements
# of a set by an arbitrarily chosen column of their embedding, prior to any
# processing. Sorting in this way is a piecewise linear function, so it is
# not fully differentiable, but when it is the first operation, that doesn't
# matter.
###############################################################################
###############################################################################


class NaiveSorter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NaiveSorter, self).__init__()
        self.emb = nn.Sequential(
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, X):
        """
        Given X: (batch, set_length, elem_dim),
        sort along the set_length axis. We do not simply use torch.sort,
        because autograd will reverse the sorting to the original order,
        and we want to show that this is non-differentiable. Otherwise,
        this would work:
        ```
            sorted, _ = torch.sort(X, dim=1)
            element_embedding = self.emb(sorted))
            return element_embedding
        ```
        Regardless, this may require further investigation.
        Small floating point differences in the torch.nn.LSTM can also
        result in the same input sequence resulting in slightly different
        outputs in the lstm encoder of a Pointer Network.

        :param X: (batch_size, set_length, elem_embedding)
        """
        # obtain sorted indices of set elements along all embedding dimensions
        # assumes zeroth dimension is batches, first is individual sets
        # third is embedded elements
        sorted_indices_all = torch.argsort(X, 1)

        # select only the sorted indices along the zeroth dimension
        # of the elements of the set
        sorted_indices_zeroth = sorted_indices_all[:, :, 0]

        # get the input in sorted order, along proper axis
        sorted_input = torch.stack(
            [X[i, sorted_indices_zeroth[i], :] for i in
             range(X.size()[0])])

        element_embedding = self.emb(sorted_input)
        return element_embedding


###############################################################################
###############################################################################
# 1. DEEP SETS (2017)
# Formalization of a simple way to obtain a permutation invariant representation
# of the input set.
# Original paper: https://arxiv.org/abs/1703.06114
# Implementation follows code provided by the author:
# https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb
###############################################################################
###############################################################################


class DeepSets(nn.Module):
    """
    Following the paper author's implementation:
    https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb
    It requires a stack of matrix multiplications, followed by e.g. mean or sum
    """

    def __init__(self, in_dim, out_dim, hidden):
        super(DeepSets, self).__init__()
        self.encode = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
            nn.ReLU()
        )

    def forward(self, X):
        element_embedding = self.encode(X)
        set_embedding = element_embedding.mean(-2)  # mean or sum
        set_embedding = self.out(set_embedding)
        return set_embedding, element_embedding


class DeepSetsPointerNetwork(nn.Module):
    """
    DeepSets encoder + PtrNet. Set representation concatenated to each
    element before passing to PtrNet.
    """

    def __init__(self,
                 elem_dims,
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False,
                 masking=True,
                 output_length=None,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(DeepSetsPointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        if embedding_by_dict:
            self.embedding = nn.Embedding(embedding_by_dict_size,
                                          embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)
        self.set_embedding = DeepSets(embedding_dim, embedding_dim, hidden_dim)
        self.encoder = PointerEncoder(embedding_dim * 2,  # concat -> times 2
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)
        self.decoder = PointerDecoder(embedding_dim * 2, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim * 2),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # inputs: (batch * seq_len, elem_dim)
        reshaped_inputs = inputs.view(batch_size * input_length, -1)

        # embedded_inputs: (batch, seq_len, embedding)
        if self.embedding_by_dict:
            reshaped_inputs = reshaped_inputs.long()
        else:
            reshaped_inputs = reshaped_inputs.float()
        embedded_inputs = self.embedding(reshaped_inputs).view(batch_size,
                                                               input_length, -1)

        # embed the entire set, perm-invar
        embedded_set, embedded_inputs = self.set_embedding(embedded_inputs)

        # juggle dimensions of the set representation to match the batch of elems
        embedded_set = embedded_set.unsqueeze(1).expand(-1,
                                                        embedded_inputs.size()[
                                                            1], -1)

        # conatenate
        embedded_inputs_and_set = torch.cat((embedded_inputs, embedded_set), 2)

        # encoder_hidden0: [(num_lstms, batch_size,  hidden),
        #                   (num_lstms, batch_size,  hidden]
        # where the length depends on number of lstms & bidir
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs_and_set)

        # encoder_outputs: (batch_size, seq_len, hidden)
        # encoder_hidden: [(num_lstms, batch_size, hidden),
        #                  (num_lstms, batch_size, hidden]
        # where the length depends on number of lstms & bidir
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs_and_set,
                                                       encoder_hidden0)

        if self.bidir:
            # last layer's h and c only, concatenated
            decoder_hidden0 = (
                torch.cat(
                    (encoder_hidden[0][-2:][0], encoder_hidden[0][-2:][1]),
                    dim=-1),
                torch.cat(
                    (encoder_hidden[1][-2:][0], encoder_hidden[1][-2:][1]),
                    dim=-1))
        else:
            # decoder_hidden0: ((batch, hidden),
            #                   (batch, hidden))
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(
            embedded_inputs_and_set,
            decoder_input0,
            decoder_hidden0,
            encoder_outputs)

        return outputs, pointers


###############################################################################
###############################################################################
# 2. SIMPLE WEIGHTED ATTENTION
# My own idea of obtaining a permutation invariant representation from a set,
# capable of handling varying length input, where we still use sum to get this
# invariance, but unlike in DeepSets, it's a weighted sum.
# However, these attention-weights only depend on each element's original
# embedding, not on the entire set (possible future improvement).
###############################################################################
###############################################################################


# class SimpleWeightedSum(nn.Module):
#     def __init__(self, in_dim, out_dim, hidden_dim, att_dim=1):
#         super(SimpleWeightedSum, self).__init__()
#         self.encode = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, out_dim)
#         )
#         self.attention = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, att_dim)
#         )
#
#     def forward(self, X):
#         element_embedding = self.encode(X)
#         att = self.attention(X)
#         att = att.squeeze(-1).unsqueeze(1)
#         set_embedding = torch.bmm(att, element_embedding)
#         return set_embedding, element_embedding


# class CustomSetEmbedder(nn.Module):
#     """
#     Simplest attention-based set encoding, for perm-invar.
#     Might need to add relu and stack.
#     """
#
#     def __init__(self, dim_input, num_outputs, dim_hidden, ln=False):
#         super(CustomSetEmbedder, self).__init__()
#         self.dim_input = dim_input
#         self.num_outputs = num_outputs
#         self.dim_hidden = dim_hidden
#
#         # self.e1 = nn.Linear(dim_input, dim_hidden)
#         self.e1 = nn.Sequential(
#             nn.Linear(dim_input, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, dim_hidden * 2),
#             nn.ReLU(),
#             nn.Linear(dim_hidden * 2, dim_hidden * 2),
#             nn.ReLU(),
#             nn.Linear(dim_hidden * 2, dim_hidden * 2),
#             nn.ReLU(),
#             nn.Linear(dim_hidden * 2, dim_hidden),
#             nn.ReLU(),
#         )
#         # self.a1 = nn.Linear(dim_input, num_outputs)
#         self.a1 = nn.Sequential(
#             nn.Linear(dim_input, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, dim_hidden * 2),
#             nn.ReLU(),
#             nn.Linear(dim_hidden * 2, dim_hidden),
#             nn.ReLU(),
#             nn.Linear(dim_hidden, num_outputs),
#             nn.ReLU()
#         )
#
#     def forward(self, X):
#         element_embedding = self.e1(X)
#         a = self.a1(X)
#         a = a.squeeze(-1).unsqueeze(1)
#         set_embedding = torch.bmm(a, element_embedding)
#         return set_embedding, element_embedding


# class CustomAttentionSetEmbedderOLD(nn.Module):
#     """
#     Simplest attention-based set encoding, for permutation invariant
#     representation and element embedding.
#     TODO: add nonlinearities and batch normalization
#     """
#
#     def __init__(self, dim_input, num_outputs, dim_hidden, ln=False):
#         super(CustomAttentionSetEmbedder, self).__init__()
#         self.dim_input = dim_input
#         self.num_outputs = num_outputs
#         self.dim_hidden = dim_hidden
#
#         # non-linearity
#         self.nonlinearity = nn.Tanh()
#
#         # layers
#         self.e1 = nn.Linear(dim_input, dim_hidden)
#         self.s1 = nn.Linear(dim_input, num_outputs)
#         self.a1 = nn.Linear(dim_hidden, dim_hidden)
#
#         # added for stacking
#         self.e2 = nn.Linear(dim_hidden, dim_hidden)
#         self.s2 = nn.Linear(dim_hidden, num_outputs)
#         self.a2 = nn.Linear(dim_hidden, dim_hidden)
#
#     def forward(self, X):
#         # X (batch, set_size, dim_input)
#         z = self.e1(X)  # z (batch, set_size, dim_hidden)
#         s = self.s1(X)  # s (batch, set_size, num_outputs=1)
#
#         # nonlinearity
#         z = self.nonlinearity(z)
#         s = self.nonlinearity(s)
#
#         # permutation invariant set representation s
#         # requires num_outputs == 1
#         s = s.squeeze(-1).unsqueeze(1)  # s (batch, 1, set_size)
#         s = torch.bmm(s, z)  # s (batch, 1, dim_hidden)
#
#         # real attention needs to use the embedded elements
#         # and the set embedding, to output a vector of len = num set elems,
#         # that should be softmaxed at the end, but also the order
#         # has to be tied to the order of the elements.
#
#         # need to slightly reshape the set embedding
#         s = self.a1(s)  # s (batch, 1, dim_hidden)
#         s = self.nonlinearity(s)
#         s = s.squeeze(1).unsqueeze(-1)  # s (batch, dim_hidden, 1)
#         a = torch.bmm(z, s)  # a (batch, set_size, 1)
#
#         # apply softmax, along the right dimension
#         # TODO: for some reason removing softmax really helps on binary sort
#         # a = torch.nn.functional.softmax(a, 1)
#
#         # apply the attention to elem embeddings?
#         # if we stack this, it will impact how much influence elements have on
#         # the next perm-invar set embedding
#         a = a.squeeze(-1)  # a (batch, set_size)
#         za = batch_apply_attention(a, z)  # za (batch, set_size, dim_hidden)
#
#         #### LAYER 2
#         # try to add a second layer within this one for now
#         z = self.e2(za)  # z2 (batch, set_size, dim_hidden)
#
#         # this is where we'd need to massage the set representation, instead of:
#         # s2 = self.s2(za)  # TODO: Consider adjusting self.s2 to be 64, 64 (not num-outputs)
#         s = self.s2(za)
#         # end
#
#         z = self.nonlinearity(z)
#         s = s.squeeze(-1).unsqueeze(1)
#         s = torch.bmm(s, z)
#         s = self.a2(s)
#         s = self.nonlinearity(s)
#         s = s.squeeze(1).unsqueeze(-1)
#         a = torch.bmm(z, s)
#         a = a.squeeze(-1)
#         za = batch_apply_attention(a, z)
#
#         #### LAYER 3
#
#         return s, za


def batch_apply_attention(A, X):
    """
    Take a batched tensor of attention weights and apply them per x.
    :param A: Attention weights to apply: (batch, seq_len)
    :param X: Embedded elements to apply attention to: (batch, seq_len, emb_dim)
    :return: a tensor of elements with applied attention:
             (batch, seq_len, emb_dim)
    """
    num_batch, set_length, emb_dim = X.size()
    attn_applied = []
    for i in range(num_batch):
        x = X[i]
        a = A[i]
        x = x.transpose(0, 1)
        r = torch.mul(a, x)
        r = r.transpose(1, 0)
        attn_applied.append(r)
    attn_applied = torch.stack(attn_applied, dim=0)
    return attn_applied


class CustomAttentionSetLayer(nn.Module):
    """
    Modularized, simplest attention-based set encoding, for permutation invariant
    representation and element embedding.
    """

    def __init__(self, dim_input, num_outputs, dim_hidden, ln=False):
        super(CustomAttentionSetLayer, self).__init__()
        self.dim_input = dim_input
        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden

        # layers
        self.e = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU()
        )
        self.s = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs),
            nn.ReLU()
        )
        self.a = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU()
        )

    def forward(self, X):
        # X (batch, set_size, dim_input)
        z = self.e(X)  # z (batch, set_size, dim_hidden)
        s = self.s(X)  # s (batch, set_size, num_outputs=1)

        # permutation invariant set representation s
        # requires num_outputs == 1
        s = s.squeeze(-1).unsqueeze(1)  # s (batch, 1, set_size)
        s = torch.bmm(s, z)  # s (batch, 1, dim_hidden)

        # need to slightly reshape the set embedding
        s = self.a(s)  # s (batch, 1, dim_hidden)
        s = s.squeeze(1).unsqueeze(-1)  # s (batch, dim_hidden, 1)
        # # TODO: consider another Linear() over z before this bmm
        a = torch.bmm(z, s)  # a (batch, set_size, 1)

        # apply softmax, along the right dimension
        # TODO: for some reason removing softmax really helps on binary sort
        # a = torch.nn.functional.softmax(a, 1)

        # apply the attention to elem embeddings?
        # if we stack this, it will impact how much influence elements have on
        # the next perm-invar set embedding
        a = a.squeeze(-1)  # a (batch, set_size)
        za = batch_apply_attention(a, z)  # za (batch, set_size, dim_hidden)

        return s, za


class CustomAttentionSetEmbedder(nn.Module):
    """
    Simplest attention-based set encoding, for permutation invariant
    representation and element embedding.
    TODO: add batch normalization
    """

    def __init__(self, dim_input, num_outputs, dim_hidden, ln=False):
        super(CustomAttentionSetEmbedder, self).__init__()
        self.dim_input = dim_input
        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden

        # layers
        self.l1 = CustomAttentionSetLayer(self.dim_input,
                                          self.num_outputs,
                                          self.dim_hidden)
        self.l2 = CustomAttentionSetLayer(self.dim_hidden,  # notice this
                                          self.num_outputs,
                                          self.dim_hidden)

    def forward(self, X):
        s, za = self.l1(X)
        s, za = self.l2(za)
        return s, za


class CustomAttentionPointerNetwork(nn.Module):
    """
    Custom set encoder + PtrNet. Set representation concatenated to each
    element before passing to PtrNet.
    """

    def __init__(self,
                 elem_dims,
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False,
                 masking=True,
                 output_length=None,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(CustomAttentionPointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        if embedding_by_dict:
            self.embedding = nn.Embedding(embedding_by_dict_size,
                                          embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)
        self.set_embedding = CustomAttentionSetEmbedder(embedding_dim, 1,
                                                        hidden_dim)
        # self.set_embedding = CustomSetEmbedder(embedding_dim, 1, hidden_dim,
        #                                                 hidden_dim)
        self.encoder = PointerEncoder(embedding_dim * 2,  # concat -> times 2
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)
        self.decoder = PointerDecoder(embedding_dim * 2, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim * 2),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # inputs: (batch * seq_len, elem_dim)
        recast_inputs = inputs.view(batch_size * input_length, -1)

        # embedded_inputs: (batch, seq_len, embedding)
        if self.embedding_by_dict:
            recast_inputs = recast_inputs.long()
        else:
            recast_inputs = recast_inputs.float()
        embedded_inputs = self.embedding(recast_inputs).view(batch_size,
                                                             input_length, -1)

        # embed the entire set, perm-invar
        embedded_set, embedded_inputs = self.set_embedding(embedded_inputs)

        # squeeze the middle dimension
        embedded_set = embedded_set.squeeze(-1)

        # juggle dimensions of the set representation to match the batch of elems
        embedded_set = embedded_set.unsqueeze(1).expand(-1,
                                                        embedded_inputs.size()[
                                                            1], -1)

        # conatenate
        embedded_inputs_and_set = torch.cat((embedded_inputs, embedded_set), 2)

        # encoder_hidden0: [(num_lstms, batch_size,  hidden),
        #                   (num_lstms, batch_size,  hidden]
        # where the length depends on number of lstms & bidir
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs_and_set)

        # encoder_outputs: (batch_size, seq_len, hidden)
        # encoder_hidden: [(num_lstms, batch_size, hidden),
        #                  (num_lstms, batch_size, hidden]
        # where the length depends on number of lstms & bidir
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs_and_set,
                                                       encoder_hidden0)

        if self.bidir:
            # last layer's h and c only, concatenated
            decoder_hidden0 = (
                torch.cat(
                    (encoder_hidden[0][-2:][0], encoder_hidden[0][-2:][1]),
                    dim=-1),
                torch.cat(
                    (encoder_hidden[1][-2:][0], encoder_hidden[1][-2:][1]),
                    dim=-1))
        else:
            # decoder_hidden0: ((batch, hidden),
            #                   (batch, hidden))
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(
            embedded_inputs_and_set,
            decoder_input0,
            decoder_hidden0,
            encoder_outputs)

        return outputs, pointers


###############################################################################
###############################################################################
# 3. Pointer Network (2015)
# One of the first model architectures applicable to set-to-sequence
# challenges (consist of a permutation-sensitive set encoder and outputs
# a reordering. Involves a modified attention mechanism, works with inputs of
# varying length. Includes an optional masking mechanism to prevent the model
# from pointing to the same element of the input sequence twice.
# Original paper: https://arxiv.org/abs/1506.03134
# Implementation follows:
# https://github.com/shirgur/PointerNet/blob/master/PointerNet.py
###############################################################################
###############################################################################


class PointerEncoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder
        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerEncoder, self).__init__()
        self.hidden_dim = hidden_dim // 2 if bidir else hidden_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir,
                            batch_first=True)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs, hidden):
        """
        Encoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        # if batch_first = True, not needed:
        # embedded_inputs = embedded_inputs.permute(1, 0, 2)
        torch.set_default_dtype(torch.float64)

        outputs, hidden = self.lstm(embedded_inputs, hidden)

        return outputs, hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-Net
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.c0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class PointerAttention(nn.Module):
    """
    Attention model for Pointer-Net.
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(PointerAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]),
                              requires_grad=False)
        self.soft = torch.nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # input: (batch, hidden)
        # context: (batch, seq_len, hidden)

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1,
                                                           context.size(1))

        # context: (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)

        # ctx: (batch, hidden_dim, seq_len)
        ctx = self.context_linear(context)

        # V: (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # att: (batch, seq_len)
        att = torch.bmm(V, torch.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        # alpha: (batch, seq_len)
        alpha = self.soft(att)

        # hidden_state: (batch, hidden)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class PointerDecoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim, masking=True,
                 output_length=None):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        """

        super(PointerDecoder, self).__init__()
        self.masking = masking
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = PointerAttention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            # x: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h, c = hidden

            # gates: (batch, hidden * 4)
            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)

            # input, forget, cell, out: (batch, hidden)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            # h_t: (batch, hidden)
            # context: (batch, seq_len, hidden)
            # mask: (batch, seq_len)
            hidden_t, output = self.att(h_t, context, torch.eq(mask, 0))
            hidden_t = torch.tanh(
                self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        output_length = input_length
        if self.output_length:
            output_length = self.output_length
        for _ in range(output_length):
            # decoder_input: (batch, embedding)
            # hidden: ((batch, hidden),
            #          (batch, hidden))
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1,
                                                                      outs.size()[
                                                                          1])).float()

            # Update mask to ignore seen indices, if masking is enabled
            if self.masking:
                mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1,
                                                                  self.embedding_dim).byte()

            # Below line aims to fixes:
            # UserWarning: indexing with dtype torch.uint8 is now deprecated,
            # please use a dtype torch.bool instead.
            embedding_mask = embedding_mask.bool()

            decoder_input = embedded_inputs[embedding_mask.data].view(
                batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class PointerNetwork(nn.Module):
    """
    Pointer-Net, with optional masking to prevent
    pointing to the same element twice (and never pointing to another).
    """

    def __init__(self,
                 elem_dims,
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False,
                 masking=True,
                 output_length=None,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        if embedding_by_dict:
            self.embedding = nn.Embedding(embedding_by_dict_size,
                                          embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)
        self.encoder = PointerEncoder(embedding_dim,
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)
        self.decoder = PointerDecoder(embedding_dim, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # inputs: (batch * seq_len, elem_dim)
        input = inputs.view(batch_size * input_length, -1)

        # embedded_inputs: (batch, seq_len, embedding)
        if self.embedding_by_dict:
            input = input.long()
        else:
            input = input.float()
        embedded_inputs = self.embedding(input).view(batch_size,
                                                     input_length, -1)

        # encoder_hidden0: [(num_lstms, batch_size,  hidden),
        #                   (num_lstms, batch_size,  hidden]
        # where the length depends on number of lstms & bidir
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)

        # encoder_outputs: (batch_size, seq_len, hidden)
        # encoder_hidden: [(num_lstms, batch_size, hidden),
        #                  (num_lstms, batch_size, hidden]
        # where the length depends on number of lstms & bidir
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs,
                                                       encoder_hidden0)

        if self.bidir:
            # last layer's h and c only, concatenated
            decoder_hidden0 = (
                torch.cat(
                    (encoder_hidden[0][-2:][0], encoder_hidden[0][-2:][1]),
                    dim=-1),
                torch.cat(
                    (encoder_hidden[1][-2:][0], encoder_hidden[1][-2:][1]),
                    dim=-1))
        else:
            # decoder_hidden0: ((batch, hidden),
            #                   (batch, hidden))
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return outputs, pointers


###############################################################################
###############################################################################
# 4. Read-Process-and-Write (2016)
# The first set-to-sequence model to obtain a permutation-invariant
# representation of the input set, consisting of a modified Pointer Network.
# Includes an optional masking mechanism to
# prevent the model from pointing to the same element of the input sequence twice.
# Original paper: https://arxiv.org/abs/1511.06391
# Implementation combines elements of the Pointer Network implementation
# with: github.com/materialsvirtuallab/megnet/blob/master/megnet/layers/readout/set2set.py
###############################################################################
###############################################################################


class RPWEncoder(nn.Module):
    """
    Encoder class for Read-Process-and-Write (RPW)
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 t_steps,
                 n_layers=1):
        """
        Initiate Encoder
        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int t_steps; number of steps of the permutation invariant lstm
        :param int n_layers; hardcoded to be 1 for simplicity for now
        """

        super(RPWEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.t_steps = t_steps
        self.n_layers = n_layers

        # increasing hidden_to_hidden to hidden_dim * 2 for q_star
        self.hidden_to_hidden = nn.Linear(hidden_dim * 2, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs):
        """
        Encoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :return: LSTMs outputs and hidden units (h, c)
        """
        # memory: (batch, seq_len, hidden)
        # is the embedded inputs
        memory = embedded_inputs

        # intialize q_star (hidden state) and c (cell state)
        # q_star: (batch, embedding * 2)
        # cell_state: (batch, embedding)
        #  no additional dimension since n_lstms = 1 in encode
        q_star, cell_state = self.init_hidden(embedded_inputs)

        def step(q_star, cell_state, memory):
            """
            Recurrence step function
            :param Tensor q_star: query vector, perm-invariant state
            :param Tensor memory: memory vector (embedded input sequence)
            :return: Tensor q_t, after each t step
            """

            ### Part 1 | Modified LSTM

            # removed the self.input_to_hidden,
            # as we're not supposed to have input in an RPW encoder lstm
            # gates: (batch, hidden * 4)
            gates = self.hidden_to_hidden(q_star)

            # input, forget, cell, out: (batch, hidden)
            input, forget, cell, out = gates.chunk(4, 1)

            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            # c_t: (batch, hidden)
            # h_t: (batch, hidden)
            c_t = (forget * cell_state) + (input * cell)
            h_t = out * torch.tanh(c_t)

            ### Part 2 | Attention

            # RPW attention section
            e_t = torch.bmm(h_t.unsqueeze(1), memory.permute(0, 2, 1))
            a_t = torch.softmax(e_t.squeeze(1), dim=1)  # softmax attention
            r_t = torch.bmm(a_t.unsqueeze(1), memory).squeeze(1)
            q_t_next = torch.cat((h_t, r_t), dim=1)

            return q_t_next, cell_state

        # perform t_steps towards permutation invariant representation
        for i in range(self.t_steps):
            q_star, cell_state = step(q_star, cell_state, memory)

        return q_star

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units
        :param Tensor embedded_inputs: The embedded input of Pointer-Net
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).repeat(batch_size, self.hidden_dim * 2)
        c0 = self.h0.unsqueeze(0).repeat(batch_size, self.hidden_dim)

        return h0, c0


class RPWPointerAttention(nn.Module):
    """
    Attention model for RPW.
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention
        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(RPWPointerAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]),
                              requires_grad=False)
        self.soft = torch.nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,
                context,
                mask):
        """
        Attention - Forward-pass
        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """
        # input: (batch, hidden)
        # context: (batch, seq_len, hidden)

        # (batch, hidden_dim, seq_len)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1,
                                                           context.size(1))

        # context: (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)

        # ctx: (batch, hidden_dim, seq_len)
        ctx = self.context_linear(context)

        # V: (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)

        # att: (batch, seq_len)
        att = torch.bmm(V, torch.tanh(inp + ctx)).squeeze(1)
        if len(att[mask]) > 0:
            att[mask] = self.inf[mask]

        # alpha: (batch, seq_len)
        alpha = self.soft(att)

        # hidden_state: (batch, hidden)
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class RPWPointerDecoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 masking=True):
        """
        Initiate Decoder
        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        :param bool masking: whether to allow masking
        """

        super(RPWPointerDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.masking = masking

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = RPWPointerAttention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, embedded_inputs,
                decoder_input,
                hidden):
        """
        Decoder - Forward-pass
        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)
        input_length = embedded_inputs.size(1)

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()

        outputs = []
        pointers = []

        def step(x, hidden, embedded_inputs_as_context):
            """
            Recurrence step function
            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :param embedded_inputs_as_context -> replaced encoder hidden states
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)
            input, forget, cell, out = gates.chunk(4, 1)

            # input, forget, cell, out: (batch, hidden)
            input = torch.sigmoid(input)
            forget = torch.sigmoid(forget)
            cell = torch.tanh(cell)
            out = torch.sigmoid(out)

            c_t = (forget * c) + (input * cell)
            h_t = out * torch.tanh(c_t)

            # Attention section
            hidden_t, output = self.att(h_t,
                                        embedded_inputs_as_context,
                                        torch.eq(mask, 0))
            hidden_t = torch.tanh(
                self.hidden_out(torch.cat((hidden_t, h_t), 1)))

            return hidden_t, c_t, output

        # Recurrence loop
        for _ in range(input_length):
            h_t, c_t, outs = step(decoder_input, hidden, embedded_inputs)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1,
                                                                      outs.size()[
                                                                          1])).float()

            # Update mask to ignore seen indices
            if self.masking:
                mask = mask * (1 - one_hot_pointers)

            # Get embedded inputs by max indices
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1,
                                                                  self.embedding_dim).byte()

            # Below line aims to fix:
            # UserWarning: indexing with dtype torch.uint8 is now deprecated,
            # please use a dtype torch.bool instead.
            embedding_mask = embedding_mask.bool()

            decoder_input = embedded_inputs[embedding_mask.data].view(
                batch_size, self.embedding_dim)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)
        pointers = torch.cat(pointers, 1)

        return (outputs, pointers), hidden


class ReadProcessWrite(nn.Module):
    """
    Read-Process-and-Write for FloatSorting and TSP (pre-embedded data).
    """

    def __init__(self,
                 elem_dim,
                 embedding_dim,
                 hidden_dim,
                 t_steps,
                 n_lstm_layers=1,
                 masking=True):
        """
        Initiate Pointer-Net
        :param int elem_dim: dimensionality of a single input elem
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int t_steps: steps to evolve perm-invar state
        :param int n_lstm_layers: lstsm in RPW encoding, for now 1
        :param bool masking: whether to allows masking of already pointed to
                             elements.
        """

        super(ReadProcessWrite, self).__init__()
        self.embedding_dim = embedding_dim
        self.masking = masking
        self.embedding = nn.Linear(elem_dim, embedding_dim)
        self.encoder = RPWEncoder(embedding_dim,
                                  hidden_dim,
                                  t_steps,
                                  n_lstm_layers)
        self.decoder = RPWPointerDecoder(embedding_dim, hidden_dim,
                                         masking=self.masking)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim),
                                        requires_grad=False)
        self.decoder_cell0 = Parameter(torch.FloatTensor(embedding_dim),
                                       requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch_size x seq_len)
               unless elements have dimensionality (e.g. in TSP).
               In FloatSorting it's nonexistent.
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # embedded_inputs: (batch, seq_len, embedding)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length,
                                                      -1)

        # main RPW Process block
        # q_star = (batch, hidden * 2)
        q_star = self.encoder(embedded_inputs)

        # TODO: Investigate why the two q_star representations are
        # TODO: ... very close but not exactly the same for 2 permutations
        # TODO: ... of the same set!
        # for i, e in enumerate(q_star[0][:100]):
        #     if e != q_star[1][i]:
        #         print("0: {:.6f} 1: {:.6g}".format(e, q_star[1][i]))

        # decoder_input0: (batch, embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # q_star is meant to be the entire context for the pointer
        # so we just have to reverse the concatenation to get both
        # hidden and cell state for the adjusted pointer lstm decoder
        decoder_hidden0 = torch.split(q_star, self.embedding_dim, dim=1)

        # RPW Write block via Pointer LSTM, adjusted
        # embedded_inputs: (batch, set_length, embedding_dim)
        # decoder_input0: (batch, embedding_dim)
        # decoder_hidden: 2-tuple of 2x (batch, embedding_dim)
        (outputs, pointers), decoder_hidden = self.decoder(embedded_inputs,
                                                           decoder_input0,
                                                           decoder_hidden0)

        return outputs, pointers


###############################################################################
###############################################################################
# 5. Set Transformer (2019)
# One of the latest set-encoding methods, that ensures permutation
# invariance, handles sets of varying lenghts. It's novel contribution
# is that it is able to encode higher-order interactions between elements.
# Original paper: https://arxiv.org/abs/1810.00825
# Implementation follows code provided by the authors:
# with: https://github.com/juho-lee/set_transformer
###############################################################################
###############################################################################


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    """
    This version currently uses SAB modules in the permutation-equivariant,
    per-element embedding, but it can also use ISAB modules there. Final
    decision will be made based on experiments.
    """

    def __init__(self, dim_input, num_outputs, dim_hidden, dim_output,
                 num_heads=8, ln=False):
        super(SetTransformer, self).__init__()
        self.dim_input = dim_input
        self.num_outputs = num_outputs
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_heads = num_heads

        self.emb = nn.Sequential(
            SAB(self.dim_input, self.dim_hidden, self.num_heads, ln=ln),
            SAB(self.dim_hidden, self.dim_hidden, self.num_heads, ln=ln)
        )
        self.enc = nn.Sequential(
            PMA(self.dim_hidden, self.num_heads, self.num_outputs, ln=ln),
            SAB(self.dim_hidden, self.dim_hidden, self.num_heads, ln=ln),
            SAB(self.dim_hidden, self.dim_hidden, self.num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, X):
        element_embedding = self.emb(X)
        set_embedding = self.enc(element_embedding)
        return set_embedding, element_embedding


class SetTransformerPointerNetwork(nn.Module):
    """
    SetTransformer encoder + PtrNet. Set representation concatenated to each
    element before passing to PtrNet.
    """

    def __init__(self,
                 elem_dims,
                 embedding_dim,
                 hidden_dim,
                 lstm_layers,
                 dropout,
                 bidir=False,
                 masking=True,
                 output_length=None,
                 embedding_by_dict=False,
                 embedding_by_dict_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(SetTransformerPointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.masking = masking
        self.output_length = output_length
        self.embedding_by_dict = embedding_by_dict
        self.embedding_by_dict_size = embedding_by_dict_size
        if embedding_by_dict:
            self.embedding = nn.Embedding(embedding_by_dict_size,
                                          embedding_dim)
        else:
            self.embedding = nn.Linear(elem_dims, embedding_dim)
        self.set_embedding = SetTransformer(embedding_dim, 1, embedding_dim,
                                            embedding_dim)
        self.encoder = PointerEncoder(embedding_dim * 2,  # concat -> times 2
                                      hidden_dim,
                                      lstm_layers,
                                      dropout,
                                      bidir)
        self.decoder = PointerDecoder(embedding_dim * 2, hidden_dim,
                                      masking=self.masking,
                                      output_length=self.output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim * 2),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, inputs):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        batch_size = inputs.size(0)
        input_length = inputs.size(1)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)

        # inputs: (batch * seq_len, elem_dim)
        reshaped_inputs = inputs.view(batch_size * input_length, -1)

        # embedded_inputs: (batch, seq_len, embedding)
        if self.embedding_by_dict:
            reshaped_inputs = reshaped_inputs.long()
        else:
            reshaped_inputs = reshaped_inputs.float()
        embedded_inputs = self.embedding(reshaped_inputs).view(batch_size,
                                                               input_length, -1)

        # embed the entire set, perm-invar
        embedded_set, embedded_inputs = self.set_embedding(embedded_inputs)

        # in set transformer, we need to squeeze the middle dimension
        embedded_set = embedded_set.squeeze(1)

        # juggle dimensions of the set representation to match the batch of elems
        embedded_set = embedded_set.unsqueeze(1).expand(-1,
                                                        embedded_inputs.size()[
                                                            1], -1)

        # conatenate
        embedded_inputs_and_set = torch.cat((embedded_inputs, embedded_set), 2)

        # encoder_hidden0: [(num_lstms, batch_size,  hidden),
        #                   (num_lstms, batch_size,  hidden]
        # where the length depends on number of lstms & bidir
        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs_and_set)

        # encoder_outputs: (batch_size, seq_len, hidden)
        # encoder_hidden: [(num_lstms, batch_size, hidden),
        #                  (num_lstms, batch_size, hidden]
        # where the length depends on number of lstms & bidir
        encoder_outputs, encoder_hidden = self.encoder(embedded_inputs_and_set,
                                                       encoder_hidden0)

        if self.bidir:
            # last layer's h and c only, concatenated
            decoder_hidden0 = (
                torch.cat(
                    (encoder_hidden[0][-2:][0], encoder_hidden[0][-2:][1]),
                    dim=-1),
                torch.cat(
                    (encoder_hidden[1][-2:][0], encoder_hidden[1][-2:][1]),
                    dim=-1))
        else:
            # decoder_hidden0: ((batch, hidden),
            #                   (batch, hidden))
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder(
            embedded_inputs_and_set,
            decoder_input0,
            decoder_hidden0,
            encoder_outputs)

        return outputs, pointers


if __name__ == '__main__':
    pass
