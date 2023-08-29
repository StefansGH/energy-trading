# File: trading_bot.py
# Author: Stefan van Rest
# Date: August 29, 2023
# Description: This module contains functions trading the energy markets

import torch
import torch.nn as nn


def trading_bot(input):
    """
    Outputs a buy/sell prediction, based on market information

    Args:
        values (list): A list of numerical values, inlcuding the following variables:
            last price change (t-1)
            current hour of the day
            current day of the week
            delivery hour of the day
            delivery day of the week
            product type "Intraday_Power_D" (yes/no)
            product type "Quarterly_Hour_Power" (yes/no)
            buy delivery area "10YDE-ENBW-----N" (yes/no)
            buy delivery area "del__10YDE-EON------1" (yes/no)
            buy delivery area "del__10YDE-RWENET---I" (yes/no)
            buy delivery area "del__10YDE-VE-------2" (yes/no)
            sell delivery area "10YDE-ENBW-----N" (yes/no)
            sell delivery area "del__10YDE-EON------1" (yes/no)
            sell delivery area "del__10YDE-RWENET---I" (yes/no)
            sell delivery area "del__10YDE-VE-------2" (yes/no)
    Returns:
        text: The trading decision
    """

    model = Model()
    model.load_state_dict(torch.load('models/model.pth'))
    model.eval()  # Set the model to evaluation mode

    input = torch.tensor(input).unsqueeze(0)
    pred = model(input)

    # trading decision using a threshold
    if pred<-0.03: 
        return "Sell!"
    elif pred>0.03:
        return "Buy!"
    else:
        return "Hold"
    


class Model(nn.Module):
    """
    A simple neural net
    """
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(15, 64) 
        self.fc2 = nn.Linear(64, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x