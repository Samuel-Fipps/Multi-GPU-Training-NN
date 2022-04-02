import torch.cuda
import torch
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from distutils import util
import matplotlib.pyplot as plt
import tensorflow as tf
from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import ray


os.environ["GRPC_FORK_SUPPORT_ENABLED"]="1"
os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1" 

#-------------------------------------------------------------------------
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
np.set_printoptions(threshold=sys.maxsize) 
torch.set_printoptions(threshold=10_000)
#-------------------------------------------------------------------------

input_data = torch.Tensor(np.load("biginputdata.npy", allow_pickle=True))
predict_data = torch.Tensor(np.load("bigpredictdata.npy", allow_pickle=True))
#testingdata_x = torch.Tensor(np.load("1testingdata_x.npy", allow_pickle=True))
#testingdata_y = torch.Tensor(np.load("1testingdata_y.npy", allow_pickle=True))

#testingdata_x = testingdata_x.type(torch.FloatTensor)
#testingdata_y = testingdata_y.type(torch.LongTensor)
input_data = input_data.type(torch.FloatTensor)
predict_data = predict_data.type(torch.LongTensor)



testingdata_x = torch.Tensor(np.load("1inputData.npy", allow_pickle=True))
testingdata_y = torch.Tensor(np.load("1predict.npy", allow_pickle=True))

testingdata_x = testingdata_x.type(torch.FloatTensor)
testingdata_y = testingdata_y.type(torch.LongTensor)

class NeuralNet(nn.Module):
    def __init__(self, l1=120, l2=84):
        super(NeuralNet, self ).__init__()
        self.fc1 = nn.Linear(248, l1).to(device)
        self.fc2 = nn.Linear(l1, l2).to(device)
        self.fc3 = nn.Linear(l2, 2).to(device)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x.to(device)





def train_cifar(config, checkpoint_dir=None, data_dir=None):
    net = NeuralNet(config["l1"], config["l2"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config["lr"],momentum=config["MO"], weight_decay =config["WD"],dampening=config["DP"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset = input_data

    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])

    valloader = val_subset


    for epoch in range(10000):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i in (range(0, len(input_data), (config["batch_size"]))):
            # get the inputs; data is a list of [inputs, labels]
            inputs = input_data[i:i+(config["batch_size"])]
            labels = predict_data[i:i+(config["batch_size"])]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i in (range(0, len(testingdata_x), (config["batch_size"]))):
            with torch.no_grad():
                inputs = testingdata_x[i:i+2]
                labels = testingdata_y[i:i+2]
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        #tune.report(loss=(val_loss / val_steps), accuracy=correct / total)
    print("Finished Training")


def test_accuracy(net, device="cpu"):
    batch_X = input_data
    batch_y = predict_data

    correct = 0
    total = 0
    with torch.no_grad():
        for data in range(5000):
            images=batch_X[data]
            labels=batch_y[data]
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total




def main(num_samples=5000, max_num_epochs=10000, gpus_per_trial=.1):
    data_dir = os.path.abspath("./data")
    #load_data(data_dir)
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 16)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 16)),
        "lr": tune.loguniform(1e-8, 1e-1),
        "WD": tune.loguniform(1e-15, 1e-5),
        "MO": tune.loguniform(1e-15, 1e-1),
        "DP": tune.loguniform(1e-15, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])


    result = tune.run(tune.with_parameters(train_cifar, data_dir=data_dir),
        #partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": .5, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler
        ,
        progress_reporter=reporter,
        )


    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = NeuralNet(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    #test_acc = test_accuracy(best_trained_model, device)
    #print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_num_epochs=10000, gpus_per_trial=0.5)

