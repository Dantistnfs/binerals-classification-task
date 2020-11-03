"""
Helper functions for model training, loading, testing etc
"""

import pickle
import tqdm
import torch
import numpy as np


def train(t_model, optimizer, loss_function, train_dataset,
          validation_dataset):
    epoch = 0
    while 1:
        epoch += 1
        train_loss = 0.0
        train_accu = 0.0
        train_length = 0
        t_model.train()  # turn on training mode
        for x, y in tqdm.tqdm(train_dataset):
            train_length += len(x[0])
            optimizer.zero_grad()
            preds = t_model(x)
            loss = loss_function(preds, y)
            loss.backward()
            optimizer.step()

            preds = 1 / (1 + torch.exp(-preds))
            train_accu += torch.max(y, 1)[1].eq(torch.max(preds,
                                                          1)[1]).sum().item()

            train_loss += loss.data * x.size(0)

        train_loss /= train_length
        train_accu /= train_length

        # calculate the validation loss for this epoch
        val_loss = 0.0
        val_accu = 0.0
        val_length = 0
        t_model.eval()  # turn on evaluation mode

        for x, y in validation_dataset:
            val_length += len(x[0])
            preds = t_model(x)
            loss = loss_function(preds, y)
            val_loss += loss.data * x.size(0)
            preds = 1 / (1 + torch.exp(-preds))
            val_accu += torch.max(y, 1)[1].eq(torch.max(preds,
                                                        1)[1]).sum().item()

        val_loss /= val_length
        val_accu /= val_length
        print(
            f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}'
        )
        print(
            f'Epoch: {epoch}, Training Accuracy: {train_accu:.4f}, Validation Accuracy: {val_accu:.4f}'
        )


def predict(t_model, test_dataset):
    test_accu = 0.0
    test_length = 0
    test_preds = []
    t_model.eval()  # turn on evaluation mode
    for x, y in test_dataset:
        test_length += len(x[0])
        preds = t_model(x)
        preds = 1 / (1 + torch.exp(-preds))
        test_preds.append(preds.data.cpu().numpy())
        test_accu += torch.max(y, 1)[1].eq(torch.max(preds, 1)[1]).sum().item()

    test_accu /= test_length
    test_preds = np.vstack(test_preds)

    print(f'Test Accuracy: {test_accu:.4f}')
    return test_preds


def save_model(model, filename):
    torch.save(model, filename)


def save_vocab(vocab, filename):
    with open(filename, 'wb') as output_f:
        pickle.dump(vocab, output_f)


def load_model(filename, **kwargs):
    model = torch.load(filename, **kwargs)
    return model


def load_vocab(filename):
    with open(filename, 'rb') as input_f:
        vocab = pickle.load(input_f)
    return vocab
