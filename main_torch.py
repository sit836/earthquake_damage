import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import torch
from model_torch import LogisticRegressionTorch
import numpy as np

from tqdm import tqdm

from constants import IN_PATH
from preprocessor import Preprocessor
from submit import make_submission
from utils import try_gpu

local_test = True

X_raw = pd.read_csv(os.path.join(IN_PATH, 'train_values.csv'), index_col='building_id')
train_labels = pd.read_csv(os.path.join(IN_PATH, 'train_labels.csv'), index_col='building_id')
y = train_labels.values.ravel()

if local_test:
    X_train_raw, X_eval_raw, y_train, y_eval = train_test_split(X_raw, y, test_size=0.25, stratify=y, random_state=123)

    ppc = Preprocessor()
    X_train = ppc.process(is_train=True, X=X_train_raw, y=y_train)
    X_eval = ppc.process(is_train=False, X=X_eval_raw, y=None)
    print(f'X_train.shape: {X_train.shape}')

    # X_train = torch.tensor(X_train,
    #                      dtype=torch.float).to(device=try_gpu())
    # X_eval = torch.tensor(X_eval,
    #                      dtype=torch.float).to(device=try_gpu())
    # y_train = torch.tensor(y_train,
    #                      dtype=torch.float).to(device=try_gpu())
    # y_eval = torch.tensor(y_eval,
    #                      dtype=torch.float).to(device=try_gpu())

    X_train = torch.tensor(X_train,
                         dtype=torch.float)
    X_eval = torch.tensor(X_eval,
                         dtype=torch.float)
    y_train = torch.tensor(y_train,
                         dtype=torch.float)
    y_eval = torch.tensor(y_eval,
                         dtype=torch.float)

    epochs = 10
    input_dim = X_train.shape[1]
    output_dim = 3
    learning_rate = 0.01

    model = LogisticRegressionTorch(input_dim, output_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    losses = []
    losses_test = []
    Iterations = []
    iter = 0
    for epoch in tqdm(range(int(epochs)), desc='Training Epochs'):
        optimizer.zero_grad()  # Setting our stored gradients equal to zero
        outputs = model(X_train)
        loss = criterion(outputs, y_train.type(torch.LongTensor))

        loss.backward()  # Computes the gradient of the given tensor w.r.t. the weights/bias

        optimizer.step()  # Updates weights and biases with the optimizer (SGD)

        iter += 1
        if iter % 10000 == 0:
            with torch.no_grad():
                # Calculating the loss and accuracy for the test dataset
                correct_test = 0
                total_test = 0
                outputs_test = torch.squeeze(model(X_eval))
                loss_test = criterion(outputs_test, y_eval)

                predicted_test = outputs_test.round().detach().numpy()
                total_test += y_eval.size(0)
                correct_test += np.sum(predicted_test == y_eval.detach().numpy())
                accuracy_test = 100 * correct_test / total_test
                losses_test.append(loss_test.item())

                # Calculating the loss and accuracy for the train dataset
                total = 0
                correct = 0
                total += y_train.size(0)
                correct += np.sum(torch.squeeze(outputs).round().detach().numpy() == y_train.detach().numpy())
                accuracy = 100 * correct / total
                losses.append(loss.item())
                Iterations.append(iter)

                print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
                print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")
