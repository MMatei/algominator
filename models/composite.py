import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MultiLayerLSTM(nn.Module):
    def __init__(self, insz = 22, outsz = 11, layers_no=2, hidden_size=128):
        super(MultiLayerLSTM, self).__init__()

        self.layers_no = layers_no
        self.hidden_size = hidden_size

        self.layers = layers = []
        self.layer0 = layer0 = nn.LSTMCell(insz, hidden_size)
        self.layers.append(layer0)

        for l in range(1, layers_no):
            new_layer = nn.LSTMCell(hidden_size, hidden_size)
            setattr(self, "layer" + str(l), new_layer)
            self.layers.append(new_layer)

        self.output = nn.Linear(hidden_size, outsz)

    def forward(self, x, hidden_state=None):
        hidden_size = self.hidden_size
        layers_no = self.layers_no
        batch_size = x.size(0)

        if hidden_state is None:
            hidden_state = []
            for l in range(layers_no):
                h = x.data.new().resize_(batch_size, hidden_size).zero_()
                c = x.data.new().resize_(batch_size, hidden_size).zero_()
                hidden_state.append((Variable(h), Variable(c)))

        new_hidden_state = []

        for l, (f, h) in enumerate(zip(self.layers, hidden_state)):
            x, c = f(x, h)
            new_hidden_state.append((x, c))

        y = F.log_softmax(self.output(x))
        return y, new_hidden_state

    def run(self, X, T):
        h, loss = None, None
        Y = []
        for s in range(0, X.size(0)):
            y, h = self.forward(X[s], h)
            Y.append(y.data.numpy())
            loss_t = F.nll_loss(y, T[s])
            loss = (loss + loss_t) if loss is not None else loss_t
        return Y, loss.data[0]

    def train_step(self, X, T, o):
        h, loss = None, None
        for s in range(0, X.size(0)):
            y, h = self.forward(X[s], h)
            loss_t = F.nll_loss(y, T[s])
            loss = (loss + loss_t) if loss is not None else loss_t
        o.zero_grad()
        loss.backward()
        o.step()
        return loss.data[0]

class Composite():
    def __init__(self):
        # load separate modules
        c1 = torch.load('multiplication.bin')
        c2 = torch.load('addition.bin')
        mem1 = torch.load('memory.bin')
        mem2 = torch.load('memory.bin')