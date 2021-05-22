import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        bs, ts, na, os = inputs.shape

        x = F.relu(self.fc1(inputs))

        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x = x[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x, h)
            hs.append(h.view(bs, na, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time

        q = self.fc2(hs)
        return q, hs
