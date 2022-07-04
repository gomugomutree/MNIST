from copy import deepcopy
import numpy as np

import torch

class Trainer():

    def __init__(self, model, optimizer, crit): # classification : Log-softmax-> crit : NLL
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _batchify(self, x, y, batch_size, random_split=True):
        
        if random_split:
            # x를 기준으로 random을 돌린 index를 받아 x, y를 index 기준으로 다시 정렬
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    # mini-batch-size model train
    def _train(self, x, y, config):
        self.model.trian()

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # initailize the gradient descent
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2:
                print('Train Iteration(%d/%d) : loss=%.4e' % (i+1, len(x), float(loss_i)))
            
            # Don't forget to detach to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validation(self, x, y, config):
        # Turn evaluation mode on
        self.model.eval()

        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print('Vaild Iteration(%d/%d) : loss=%.4e' % (i+1, len(x), float(loss_i)))
            
            total_loss += float(loss_i)
        
        return total_loss / len(x)


