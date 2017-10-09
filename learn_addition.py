from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def get_batch(n=256, max_digits=10):

    x = torch.FloatTensor(max_digits, n, 2 * 10).zero_()
    t = torch.LongTensor(max_digits, n).zero_()

    a = torch.LongTensor(n).random_(10**9)
    b = torch.LongTensor(n).random_(10**9)
    d = a + b

    for i in range(max_digits):
        x[i].scatter_(1, (a % 10).unsqueeze(1), 1.)
        x[i].scatter_(1, (b % 10 + 10).unsqueeze(1), 1.)
        t[i].copy_(d % 10)

        a.div_(10)
        b.div_(10)
        d.div_(10)

    return x, t

class MultiLayerLSTM(nn.Module):
    def __init__(self, max_digits=10, layers_no=2, hidden_size=128):
        super(MultiLayerLSTM, self).__init__()

        self.layers_no = layers_no
        self.hidden_size = hidden_size

        self.layers = layers = []
        self.layer0 = layer0 = nn.LSTMCell(10 * 2, hidden_size)
        self.layers.append(layer0)

        for l in range(1, layers_no):
            new_layer = nn.LSTMCell(hidden_size, hidden_size)
            setattr(self, "layer" + str(l), new_layer)
            self.layers.append(new_layer)

        self.output = nn.Linear(hidden_size, 10)

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



def train():
    batch_size = 1024
    max_digits = 15
    epochs_no = 1000
    use_cuda = False

    m = MultiLayerLSTM(max_digits=max_digits, layers_no=1, hidden_size=128)
    # o = optim.Adam(m.parameters(), lr=.001)
    o = optim.RMSprop(m.parameters())

    if use_cuda:
        m.cuda()

    best_loss = None

    for e in range(epochs_no):
        x, t = get_batch(n=batch_size, max_digits=max_digits)

        if use_cuda:
            x = x.cuda()
            t = t.cuda()

        h, loss, acc = None, None, 0
        for d in range(max_digits):
            y, h = m(Variable(x[d]), h)
            loss_t = F.nll_loss(y, Variable(t[d]))
            loss = (loss + loss_t) if loss is not None else loss_t

            _, digits = y.data.max(1)
            n_correct = (digits == t[d]).long().sum()
            acc += n_correct / batch_size
        acc /= max_digits

        o.zero_grad()
        loss.backward()
        o.step()

        _loss = loss.data[0]
        if best_loss is None or best_loss > _loss:
            best_loss = _loss

        print("Step {:d}, Acc = {:2.2f}%, Loss = {:f}, Best loss = {:f} {:s}".format(
            e, acc*100, _loss, best_loss, "" if _loss > best_loss else "!!!!!"
        ))

if __name__ == "__main__":
    train()
