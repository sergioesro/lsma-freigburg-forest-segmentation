import torch
from torch import nn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from torch.utils.data import Dataset, DataLoader

from src.utils.representation_tools import plot_training_process

import numpy as np



def train(X, model, loss_fn, optimizer, epochs=3, normalize=True, save=False, name=None, plot=False):
    '''
    Train method focused in how hypercubes are
    '''
    # device = get_device()
    device='cuda'
    dataloader = DataLoader(HypercubeDataset(X), batch_size=32, shuffle=True)
    train_losses = []
    # X = torch.from_numpy(x_norm)
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.squeeze().to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images.float())
            
            # Reshape labels if necessary
            # labels = labels.view(-1)

            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update the running loss
            running_loss += loss.item()
        print(f'Loss {running_loss / len(HypercubeDataset(X))}')
    # Print the average loss for the epoch
        epoch_loss = running_loss / len(HypercubeDataset(X))
        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")
    if plot:
        plot_training_process(train_losses, save=True, name=name)
    if save and name is not None:
        print("Saving model, better loss obtained!")
        torch.save(model.state_dict(), f"src/datasets/neural_models/{name}.pt")

def train_kfold(X, model, loss_fn, optimizer, epochs=1, k_folds=3, normalize=True, save=False, name=None):
    y = X[-1,:]
    X = X[0:9,:]
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    y = torch.from_numpy(LabelBinarizer().fit_transform(y))
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=1)
    # Train and evaluate model using K-Fold cross-validation
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in kfold.split(X.T):
        X_train, X_test = X[:, train_index], X[:, test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(np.argmax(y_test, axis=1), dtype=torch.long)

        # Train model
        for epoch in range(epochs):
            print(f'### Training epoch {epoch+1}/{epochs} ###')
            optimizer.zero_grad()
            output = model(X_train_tensor.T)
            loss = loss_fn(output, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Evaluate model
        with torch.no_grad():
            output = model(X_test_tensor.T)
            predictions = torch.argmax(output, axis=1)
            accuracy = (predictions == y_test_tensor).float().mean().item()
            precision = precision_score(y_test_tensor, predictions, average=None, zero_division = 0, labels = [0,1,2,3,4])
            recall = recall_score(y_test_tensor, predictions, average=None, zero_division = 0, labels = [0,1,2,3,4])
            f1 = f1_score(y_test_tensor, predictions, average=None, zero_division = 0, labels = [0,1,2,3,4])

            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

    # Compute mean and standard deviation of accuracy scores
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)


    print('Accuracy: {:.2f}%'.format(mean_accuracy*100))
    print('Precision: {:.2f}%'.format(mean_precision*100))
    print('Recall: {:.2f}%'.format(mean_recall*100))
    print('f1: {:.2f}%'.format(mean_f1*100))


def train_cnn(X, model, loss_fn, optimizer, epochs=3, normalize=True, save=False, name=None):
    for epoch in range(epochs):
        running_loss = 0.0
        device='cpu'
        dataloader =  DataLoader(HypercubeCNNDataset(X), batch_size=4, shuffle=True, num_workers=2)
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # format inputs and labels
            inputs = inputs.reshape(1, inputs.shape[3],  inputs.shape[1], inputs.shape[2])
            # labels = np.reshape(labels,(1,(inputs.shape[1]*inputs.shape[2])))
            labels = torch.tensor(labels, dtype=torch.long)

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = loss_fn(outputs.flatten(), labels.flatten().float())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs} | Loss: {running_loss/11000800}')
        running_loss = 0.0
    if save and name is not None:
        print("Saving model, better loss obtained!")
        torch.save(model.state_dict(), f"src/datasets/neural_models/{name}.pt")

def test(X, X_shape, model, loss_fn, normalize=True, load=False, name=None, generate_image=False):
    '''
    Base train method for the hypercubes
    '''
    y_pred = np.zeros((1, X.shape[1]))
    if load and name is not None:
        model.load_state_dict(torch.load(f"src/datasets/neural_models/{name}.pt"))
    device = get_device()
    device = "cpu"
    size = X.shape[1]
    y = torch.from_numpy(LabelBinarizer().fit_transform(X[-1, :]))
    if normalize:
        x_norm = normalize_standard(X[:-1, :])
    else:
        x_norm = X[:-1, :]
    X = torch.from_numpy(normalize_standard(x_norm))
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in range(size):
            x, gt = X[:, i].to(device), y[i].to(device)
            pred = model(x.float())
            test_loss += loss_fn(pred, gt.float()).item()
            correct += (pred.argmax() == gt.argmax()).type(torch.float).sum().item()
            if generate_image:
                y_pred[0, i] = float(pred.argmax())
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    if generate_image:
        y_out = np.reshape(y_pred, (X_shape[0],X_shape[1]))
        return correct, test_loss, y_out
    return correct, test_loss, y_pred


def test_CNN(X, model, loss_fn, normalize=True, load=False, name=None, generate_image=False):
    dataloader =  DataLoader(HypercubeCNNDataset(X), batch_size=4, shuffle=True, num_workers=2)
    device = get_device()
    if load and name is not None:
        model.load_state_dict(torch.load(f"src/datasets/neural_models/{name}.pt"))
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients

            # format inputs and labels
            inputs = inputs.reshape(1, inputs.shape[3],  inputs.shape[1], inputs.shape[2])
            # labels = np.reshape(labels,(1,(inputs.shape[1]*inputs.shape[2])))
            labels = torch.tensor(labels, dtype=torch.long)

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = loss_fn(outputs.flatten(), labels.flatten().float())
            test_loss += loss.item()
    y_pred = outputs.reshape(outputs, (X[0].shape[0], X[0].shape[1]))
    return y_pred


def get_device():
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using {device} device")
    return device


def initialize_network_parameters(model, lr=1e-3):
    '''
    Simple initialization for the model where we define the loss and the optimizer
    '''
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return loss_fn, optimizer


# TODO: This should be in a different "package"
def normalize_standard(x):
    '''
    Given a np array normalizes its values in range of 0 and 1
    '''
    x_norm = (x-np.min(x))/(np.max(x)-np.min(x))
    return x_norm


class HypercubeDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return self.images.shape[1]

    def __getitem__(self, index):
        image = self.images[:-1, index]  # Extract image without the last layer
        label = self.images[-1, index]   # Extract the last layer as the label
        
        image = torch.from_numpy(normalize_standard(image))
        lab = np.zeros((5,1))
        lab[int(label)] = 1.0

        return image, lab
    

class HypercubeCNNDataset(Dataset):
    def __init__(self, hypercubes, transform=None):
        self.hypercubes = hypercubes
        self.transform = transform

    def __len__(self):
        return len(self.hypercubes)

    def __getitem__(self, index):
        hypercube = self.hypercubes[index][:,:,:-1]
        hypercube = normalize_standard(hypercube)
        # hypercube = np.resize(hypercube, (483,880,9))
        hypercube = torch.from_numpy(hypercube)
        labels = self.hypercubes[index][:,:,-1]
        # labels = np.resize(labels, (483,880))
        return hypercube, labels


