import torch # neural network library
import torch.nn as nn # neural network module
import torch.optim as optim # optimizer
import torch.nn.functional as F # activation functions
import os # to save the model

class Linear_QNet(nn.Module): # inherit from nn.Module
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() # inherit from nn.Module
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr) # The way the adam optimizer works is that it takes the parameters of the model and updates them based on the loss function
        self.criterion = nn.MSELoss() # mean squared error: (y - y_pred)^2

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1: # if the state is a vector, we need to add a dimension to it to make it a matrix of size (1, x) instead of a vector of size (x, )
            # (1, x)
            state = torch.unsqueeze(state, 0) # unsqueeze adds a dimension to the tensor
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state. Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # Q values are the values of each action for the current state
        pred = self.model(state) # pass the state to the model to get the predicted Q values

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


