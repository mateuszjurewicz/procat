import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter


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

class PointerOfferEmbedder(nn.Module):
    """
    Embeds offers after the words have already been embedded.
    At first through a simple RNN, maybe later we'll plug in a transformer here.
    """
    def __init__(self,
                 word_embedding_dim,
                 offer_embedding_dim,
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

        super(PointerOfferEmbedder, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.offer_embedding_dim = offer_embedding_dim // 2 if bidir else offer_embedding_dim
        self.n_layers = n_layers * 2 if bidir else n_layers
        self.bidir = bidir
        self.offer_rnn = nn.GRU(
            input_size=self.word_embedding_dim,
            hidden_size=self.offer_embedding_dim,
            num_layers=self.n_layers,
            dropout=dropout,
            bidirectional=bidir,
            batch_first=True)

    def forward(self, X_words_embedded):

        batch_hidden = []
        for catalog in X_words_embedded:
            catalog = catalog.float()
            _, hidden_states_from_layers = self.offer_rnn(catalog)
            catalog_hidden = hidden_states_from_layers[-1]
            batch_hidden.append(catalog_hidden)
        batch_hidden = torch.stack(batch_hidden, dim=0)
        return batch_hidden


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


class PointerCatalogNetwork(nn.Module):
    """
    Pointer-Net, with optional masking to prevent
    pointing to the same element twice (and never pointing to another).
    """

    def __init__(self,
                 word_embedding_dim,
                 offer_embedding_dim,
                 hidden_dim,
                 offer_rnn_layers,
                 catalog_rnn_layers,
                 dropout_offers,
                 dropout_catalogs,
                 bidir_offers=False,
                 bidir_catalogs=False,
                 masking=True,
                 output_length=None,
                 vocab_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerCatalogNetwork, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.offer_embedding_dim = offer_embedding_dim
        self.hidden_dim = hidden_dim
        self.bidir_offers = bidir_offers
        self.bidir_catalogs = bidir_catalogs
        self.masking = masking
        self.output_length = output_length
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embedding_dim)
        self.offer_embedding = PointerOfferEmbedder(word_embedding_dim,
                                                    offer_embedding_dim,
                                                    offer_rnn_layers,
                                                    dropout_offers,
                                                    bidir_offers)
        self.encoder = PointerEncoder(offer_embedding_dim,
                                      hidden_dim,
                                      catalog_rnn_layers,
                                      dropout_catalogs,
                                      bidir_catalogs)
        self.decoder = PointerDecoder(offer_embedding_dim, hidden_dim,
                                      masking=masking,
                                      output_length=output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(offer_embedding_dim),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, X):
        """
        PointerNet - Forward-pass
        :param Tensor X: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        num_catalogs_per_batch = X.size(0)
        num_offers_per_catalog = X.size(1)
        num_words_per_offer = X.size(2)

        # decoder_input0: (batch,  offer_embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(num_catalogs_per_batch, -1)

        # cast to long for the embedding layer
        X = X.long()

        # embed on word token level (batch, offers, words, word_emb)
        embedded_words = self.word_embedding(X)

        # embed on offer level (batch, offers, offer_emb) [unless bidir]
        embedded_offers = self.offer_embedding(embedded_words)

        # encoder_hidden0: [(num_lstms, batch_size,  hidden),
        #                   (num_lstms, batch_size,  hidden]
        # where the length depends on number of lstms & bidir
        encoder_hidden0 = self.encoder.init_hidden(embedded_offers)

        # encoder_outputs: (batch_size, seq_len, hidden)
        # encoder_hidden: [(num_lstms, batch_size, hidden),
        #                  (num_lstms, batch_size, hidden]
        # where the length depends on number of lstms & bidir
        encoder_outputs, encoder_hidden = self.encoder(embedded_offers,
                                                       encoder_hidden0)

        if self.bidir_catalogs:
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
        (outputs, pointers), decoder_hidden = self.decoder(embedded_offers,
                                                           decoder_input0,
                                                           decoder_hidden0,
                                                           encoder_outputs)

        return outputs, pointers


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
            nn.Linear(hidden * 2, out_dim),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Linear(out_dim, hidden),
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


class DeepSetsPointerCatalogNetwork(nn.Module):
    """
    DeepSets encoder + PtrNet. Set representation concatenated to each
    element before passing to PtrNet.
    """

    def __init__(self,
                 word_embedding_dim,
                 offer_embedding_dim,
                 hidden_dim,
                 offer_rnn_layers,
                 catalog_rnn_layers,
                 dropout_offers,
                 dropout_catalogs,
                 bidir_offers=False,
                 bidir_catalogs=False,
                 masking=True,
                 output_length=None,
                 vocab_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(DeepSetsPointerCatalogNetwork, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.offer_embedding_dim = offer_embedding_dim
        self.hidden_dim = hidden_dim
        self.bidir_offers = bidir_offers
        self.bidir_catalogs = bidir_catalogs
        self.masking = masking
        self.output_length = output_length
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embedding_dim)
        self.offer_embedding = PointerOfferEmbedder(word_embedding_dim,
                                                    offer_embedding_dim,
                                                    offer_rnn_layers,
                                                    dropout_offers,
                                                    bidir_offers)
        self.set_embedding = DeepSets(offer_embedding_dim, offer_embedding_dim, hidden_dim)
        self.encoder = PointerEncoder(offer_embedding_dim * 2,
                                      hidden_dim,
                                      catalog_rnn_layers,
                                      dropout_catalogs,
                                      bidir_catalogs)
        self.decoder = PointerDecoder(offer_embedding_dim * 2, hidden_dim,
                                      masking=masking,
                                      output_length=output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(offer_embedding_dim * 2),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, X):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        num_catalogs_per_batch = X.size(0)
        num_offers_per_catalog = X.size(1)
        num_words_per_offer = X.size(2)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(
            num_catalogs_per_batch, -1)

        # cast to long for the embedding layer
        X = X.long()

        # embed on word token level (batch, offers, words, word_emb)
        embedded_words = self.word_embedding(X)

        # embed on offer level (batch, offers, offer_emb) [unless bidir]
        embedded_offers = self.offer_embedding(embedded_words)

        # embed the entire set, perm-invar
        embedded_set, embedded_offers = self.set_embedding(embedded_offers)

        # juggle dimensions of the set representation to match the batch of elems
        embedded_set = embedded_set.unsqueeze(1).expand(-1,
                                                        embedded_offers.size()[
                                                            1], -1)

        # conatenate
        embedded_inputs_and_set = torch.cat((embedded_offers, embedded_set), 2)

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

        if self.bidir_catalogs:
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


class SetTransformerPointerCatalogNetwork(nn.Module):
    """
    SetTransformer + Pointer for catalogs.
    """

    def __init__(self,
                 word_embedding_dim,
                 offer_embedding_dim,
                 hidden_dim,
                 offer_rnn_layers,
                 catalog_rnn_layers,
                 dropout_offers,
                 dropout_catalogs,
                 bidir_offers=False,
                 bidir_catalogs=False,
                 masking=True,
                 output_length=None,
                 vocab_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(SetTransformerPointerCatalogNetwork, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.offer_embedding_dim = offer_embedding_dim
        self.hidden_dim = hidden_dim
        self.bidir_offers = bidir_offers
        self.bidir_catalogs = bidir_catalogs
        self.masking = masking
        self.output_length = output_length
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embedding_dim)
        self.offer_embedding = PointerOfferEmbedder(word_embedding_dim,
                                                    offer_embedding_dim,
                                                    offer_rnn_layers,
                                                    dropout_offers,
                                                    bidir_offers)
        self.set_embedding = SetTransformer(offer_embedding_dim, 1,
                                            offer_embedding_dim,
                                            offer_embedding_dim)
        self.encoder = PointerEncoder(offer_embedding_dim * 2,
                                      hidden_dim,
                                      catalog_rnn_layers,
                                      dropout_catalogs,
                                      bidir_catalogs)
        self.decoder = PointerDecoder(offer_embedding_dim * 2, hidden_dim,
                                      masking=masking,
                                      output_length=output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(offer_embedding_dim * 2),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, X):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        num_catalogs_per_batch = X.size(0)
        num_offers_per_catalog = X.size(1)
        num_words_per_offer = X.size(2)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(
            num_catalogs_per_batch, -1)

        # cast to long for the embedding layer
        X = X.long()

        # embed on word token level (batch, offers, words, word_emb)
        embedded_words = self.word_embedding(X)

        # embed on offer level (batch, offers, offer_emb) [unless bidir]
        embedded_offers = self.offer_embedding(embedded_words)

        # embed the entire set, perm-invar
        embedded_set, embedded_offers = self.set_embedding(embedded_offers)

        # in set transformer, we need to squeeze the middle dimension
        embedded_set = embedded_set.squeeze(1)

        # juggle dimensions of the set representation to match the batch of elems
        embedded_set = embedded_set.unsqueeze(1).expand(-1,
                                                        embedded_offers.size()[
                                                            1], -1)

        # conatenate
        embedded_inputs_and_set = torch.cat((embedded_offers, embedded_set), 2)

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

        if self.bidir_catalogs:
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
        self.l3 = CustomAttentionSetLayer(self.dim_hidden,  # notice this
                                          self.num_outputs,
                                          self.dim_hidden)
        self.l4 = CustomAttentionSetLayer(self.dim_hidden,  # notice this
                                          self.num_outputs,
                                          self.dim_hidden)

    def forward(self, X):
        s, za = self.l1(X)
        s, za = self.l2(za)
        s, za = self.l3(za)
        s, za = self.l4(za)
        return s, za


class CustomAttentionPointerCatalogNetwork(nn.Module):
    """
    SetTransformer + Pointer for catalogs.
    """

    def __init__(self,
                 word_embedding_dim,
                 offer_embedding_dim,
                 hidden_dim,
                 offer_rnn_layers,
                 catalog_rnn_layers,
                 dropout_offers,
                 dropout_catalogs,
                 bidir_offers=False,
                 bidir_catalogs=False,
                 masking=True,
                 output_length=None,
                 vocab_size=None):
        """
        Initiate Pointer-Net
        :param int elem_dims: dimensions of each invdividual set element
        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(CustomAttentionPointerCatalogNetwork, self).__init__()
        self.word_embedding_dim = word_embedding_dim
        self.offer_embedding_dim = offer_embedding_dim
        self.hidden_dim = hidden_dim
        self.bidir_offers = bidir_offers
        self.bidir_catalogs = bidir_catalogs
        self.masking = masking
        self.output_length = output_length
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size,
                                           word_embedding_dim)
        self.offer_embedding = PointerOfferEmbedder(word_embedding_dim,
                                                    offer_embedding_dim,
                                                    offer_rnn_layers,
                                                    dropout_offers,
                                                    bidir_offers)
        self.set_embedding = CustomAttentionSetEmbedder(offer_embedding_dim, 1,
                                                        offer_embedding_dim)
        self.encoder = PointerEncoder(offer_embedding_dim * 2,
                                      hidden_dim,
                                      catalog_rnn_layers,
                                      dropout_catalogs,
                                      bidir_catalogs)
        self.decoder = PointerDecoder(offer_embedding_dim * 2, hidden_dim,
                                      masking=masking,
                                      output_length=output_length)
        self.decoder_input0 = Parameter(torch.FloatTensor(offer_embedding_dim * 2),
                                        requires_grad=False)

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)

    def forward(self, X):
        """
        PointerNet - Forward-pass
        :param Tensor inputs: Input sequence (batch x seq_len x elem_dim)
        :return: Pointers probabilities and indices
        """
        # inputs: (batch x seq_len x elem_dim)
        # for TSP in 2D, elem_dim = 2
        num_catalogs_per_batch = X.size(0)
        num_offers_per_catalog = X.size(1)
        num_words_per_offer = X.size(2)

        # decoder_input0: (batch,  embedding)
        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(
            num_catalogs_per_batch, -1)

        # cast to long for the embedding layer
        X = X.long()

        # embed on word token level (batch, offers, words, word_emb)
        embedded_words = self.word_embedding(X)

        # embed on offer level (batch, offers, offer_emb) [unless bidir]
        embedded_offers = self.offer_embedding(embedded_words)

        # embed the entire set, perm-invar
        embedded_set, embedded_offers = self.set_embedding(embedded_offers)

        # we need to squeeze the last dimension
        embedded_set = embedded_set.squeeze(-1)

        # juggle dimensions of the set representation to match the batch of elems
        embedded_set = embedded_set.unsqueeze(1).expand(-1,
                                                        embedded_offers.size()[
                                                            1], -1)

        # conatenate
        embedded_inputs_and_set = torch.cat((embedded_offers, embedded_set), 2)

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

        if self.bidir_catalogs:
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