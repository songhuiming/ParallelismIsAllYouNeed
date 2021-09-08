import torch
import torch.nn as nn

class ModelArch(nn.Module):
    def __init__(self, hugg_model, args):
        super(ModelArch, self).__init__()
        self.hugg_model = hugg_model
        self.args = args
        if self.args['model_type'] == 'xlm':
            self.dropout1 = nn.Dropout(0.4)
        elif self.args['model_type'] in ('bert', 'xlmr'):
            self.dropout1 = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        if self.args['model_type'] == 'xlm':
            self.fc1 = nn.Linear(2048, 128)
        elif self.args['model_type'] in ('bert', 'xlmr'):
            self.fc1 = nn.Linear(768, 256)
        self.dropout2 = nn.Dropout(0.1)
        if self.args['model_type'] == 'xlm':
            self.fc2 = nn.Linear(128, 2)
        elif self.args['model_type'] in ('bert', 'xlmr'):
            self.fc2 = nn.Linear(256, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward step
    def forward(self, sent_id, mask):
        if self.args['model_type'] == 'xlm':
            cls_hs = self.hugg_model(sent_id, attention_mask=mask)[0]
            x = self.dropout1(cls_hs[:, 0, :])
        elif self.args['model_type'] in ('bert', 'xlmr'):
            _, cls_hs = self.hugg_model(sent_id, attention_mask=mask)
            x = self.dropout1(cls_hs)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        # output layer
        x = self.fc2(x)
        # apply softmax activation
        x = self.softmax(x)
        return x
