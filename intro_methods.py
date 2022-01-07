from numpy.core.fromnumeric import squeeze
import torch
import numpy as np
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt




tr_data = torchvision.datasets.FashionMNIST(root= './data/',
                             train=True,download=True,
                             transform= torchvision.transforms.ToTensor())

tst_data = torchvision.datasets.FashionMNIST(root= './data/',
                             train=False,download=True,
                             transform= torchvision.transforms.ToTensor())
ratio = 80
data_divide = (ratio*tr_data.data.shape[0])//100
batch_size = 32
tr_data.data,tst_data.data = tr_data.data.float(),tst_data.data.float()
X_train,Y_train = tr_data.data[:data_divide],tr_data.targets[:data_divide]
X_valid,Y_valid = tr_data.data[data_divide:],tr_data.targets[data_divide:]




def Forward():
    model = nn.Sequential(torch.nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(in_features= 784,out_features=256),
    nn.ReLU(),nn.Linear(in_features= 256,out_features=128),
    nn.ReLU(),nn.Linear(in_features= 128,out_features=10))
    return model

def Compute_loss_accuracy_predictions(X,y,Test_label = False):
    ypred = network(X)        
    loss = CE_loss(ypred,y)
    ypred = torch.softmax(ypred,dim =1)
    if(Test_label ==  True):
        predictions = torch.argmax(ypred, dim=1)
        accuracy = torch.sum(torch.argmax(ypred, dim=1) == y).item() / X.shape[0]
        return predictions,loss,accuracy
    else:
        accuracy = torch.sum(torch.argmax(ypred, dim=1) == y).item() / X.shape[0]
        return loss, accuracy


metrics = {"training_loss":[],"training_accuracy":[],"valid_loss":[],"valid_accuracy":[]}
network = Forward()
num_epochs = 20
lr = 0.001
CE_loss = nn.CrossEntropyLoss()
opt = torch.optim.Adam(network.parameters(), lr=lr)

batches = int(np.ceil(X_train.shape[0]/batch_size))

for epoch in range(num_epochs):
    for batch in range(batches):
        opt.zero_grad()
        mini_train_X,mini_train_Y = X_train[batch*batch_size:(batch+1)*batch_size],Y_train[batch*batch_size:(batch+1)*batch_size]
        mini_valid_X,mini_valid_Y = X_valid[batch*batch_size:(batch+1)*batch_size],Y_valid[batch*batch_size:(batch+1)*batch_size]
        #images = torch.squeeze(images)
        ypred = network(mini_train_X)
        loss = CE_loss(ypred,mini_train_Y)
        loss.backward()
        opt.step()
        # Computation 
    

    l,acc = Compute_loss_accuracy_predictions(X_train,Y_train)
    metrics["training_loss"].append(l)
    metrics["training_accuracy"].append(acc)
    l,acc = Compute_loss_accuracy_predictions(X_valid,Y_valid)
    metrics["valid_loss"].append(l)
    metrics["valid_accuracy"].append(acc)
    print(f'Training loss {metrics["training_loss"][-1]} and accuracy {metrics["training_accuracy"][-1]}')
    print(f'Valid loss {metrics["valid_loss"][-1]} and accuracy {metrics["valid_accuracy"][-1]}')

fig1 = plt.figure()
plt.plot(range(len(metrics["valid_accuracy"])),metrics["valid_accuracy"])
plt.plot(range(len(metrics["training_accuracy"])),metrics["training_accuracy"])
plt.legend(['valid_accuracy','train_accuracy'])
plt.show()

fig2 = plt.figure()
plt.plot(range(len(metrics["valid_loss"])),metrics["valid_loss"])
plt.plot(range(len(metrics["training_loss"])),metrics["training_loss"])
plt.legend(['valid_loss','train_loss'])
plt.show()



