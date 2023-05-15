import torch
from torch import nn
from sklearn.preprocessing import LabelBinarizer


def train(X, model, loss_fn, optimizer, save=False, name=None):
    device = get_device()
    size = X.shape[1]
    y = torch.from_numpy(LabelBinarizer().fit_transform(X[-1, :]))
    X = torch.from_numpy(X)
    model.train()
    for i in range(size):
        x, gt = X[:-1, i].to(device), y[i].to(device)
        # Compute prediction error
        pred = model(x.float())
        loss = loss_fn(pred, gt.float())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100000 == 0:
            loss, current = loss.item(), (i)
            print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    if save and name is not None:
        torch.save(model.state_dict(), f"src/datasets/neural_models/{name}.pt")


def test(X, model, loss_fn, load=False, name=None):
    if load and name is not None:
        model.load_state_dict(torch.load(f"src/datasets/neural_models/{name}.pt"))
    device = get_device()
    test_acc = []
    test_losses = []    
    size = X.shape[1]
    y = torch.from_numpy(LabelBinarizer().fit_transform(X[-1, :]))
    X = torch.from_numpy(X)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in range(size):
            x, gt = X[:-1, i].to(device), y[i].to(device)
            pred = model(x.float())
            test_loss += loss_fn(pred, gt.float()).item()
            correct += (pred.argmax() == gt.argmax()).type(torch.float).sum().item()
    correct /= size
    test_acc.append(correct)
    test_losses.append(test_loss)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_acc, test_loss


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device

def initialize_network_parameters(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return loss_fn, optimizer