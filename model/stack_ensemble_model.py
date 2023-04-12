from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import torch
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import BCELoss
from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_
import time
import numpy as np
from sklearn.metrics import accuracy_score



# The meta learner: a 3-layer MLP
class FraudMLP(Module):
    def __init__(self, n_inputs):
        super(FraudMLP, self).__init__()
        # First hidden layer
        self.hidden1 = Linear(n_inputs, 500)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # Second hidden layer
        self.hidden2 = Linear(500, 100)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = ReLU()
        # Third hidden layer
        self.hidden3 = Linear(100,1)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()

    def forward(self, X):
        #Input to the first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # Second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # Third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)

        return X

def train_model(train_dl, model, epochs=100, lr=0.001, momentum=0.9, save_path='best_model.pth'):
    # Define your optimisation function for reducing loss when weights are calculated
    # and propogated through the network
    start = time.time()
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    loss = 0.0
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)
        model.train()
        # Iterate through training data loader
        for i, (inputs, targets) in enumerate(train_dl):
            optimizer.zero_grad()
            print(inputs.shape)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data,1) #Get the class labels
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        torch.save(model, save_path)
    time_delta = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_delta // 60, time_delta % 60
    ))

    return model
#%%
import math
def evaluate_model(test_dl, model, beta=1.0):
    preds = []
    actuals = []
    for (i, (inputs, targets)) in enumerate(test_dl):
        #Evaluate the model on the test set
        yhat = model(inputs)
        #Retrieve a numpy weights array
        yhat = yhat.detach().numpy()
        # Extract the weights using detach to get the numerical values in an ndarray, instead of tensor
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        yhat = yhat.round()
        # Store the predictions in the empty lists initialised at the start of the class
        preds.append(yhat)
        actuals.append(actual)

    # Stack the predictions and actual arrays vertically
    preds, actuals = np.vstack(preds), np.vstack(actuals)
    #Calculate metrics
    cm = confusion_matrix(actuals, preds)
    print(cm)
    # Get descriptions of tp, tn, fp, fn
    tn, fp, fn, tp = cm.ravel()
    total = sum(cm.ravel())

    metrics = {
        'accuracy': accuracy_score(actuals, preds),
        'AU_ROC': roc_auc_score(actuals, preds),
        'f1_score': f1_score(actuals, preds),
        'average_precision_score': average_precision_score(actuals, preds),
        'f_beta': ((1+beta**2) * precision_score(actuals, preds) * recall_score(actuals, preds)) / (beta**2 * precision_score(actuals, preds) + recall_score(actuals, preds)),
        'matthews_correlation_coefficient': (tp*tn - fp*fn) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)),
        'precision': precision_score(actuals, preds),
        'recall': recall_score(actuals, preds),
        'true_positive_rate_TPR':recall_score(actuals, preds),
        'false_positive_rate_FPR':fp / (fp + tn) ,
        'false_discovery_rate': fp / (fp +tp),
        'false_negative_rate': fn / (fn + tp) ,
        'negative_predictive_value': tn / (tn+fn),
        'misclassification_error_rate': (fp+fn)/total ,
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        #'confusion_matrix': confusion_matrix(actuals, preds),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn
    }
    return metrics, preds, actuals

def predict(row, model):
    row = Tensor([row])
    yhat = model(row)
    # Get numpy array
    yhat = yhat.detach().numpy()
    return yhat
