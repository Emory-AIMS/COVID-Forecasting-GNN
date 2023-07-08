import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression

class Linear(nn.Module): 
    def __init__(self, args, data):
        super().__init__()
        self.cuda = args.cuda
        self.window = args.window
        self.horizon = args.horizon

    def forward(self, x):
        y_vals = x.permute(0, 2, 1).detach().cpu().numpy()
        x_vals = np.array(list(range(self.window))).reshape((-1, 1))
        x_pred = [[self.window + self.horizon - 1]]
        out = []
        for batch in y_vals:
            pred = []
            for county in batch:
                model = LinearRegression()
                model.fit(x_vals, county)
                pred.append(model.predict(x_pred))
            out.append(pred)
        out = torch.squeeze(torch.tensor(out))
        if self.cuda:
            out = out.cuda()                         
        return out, None
