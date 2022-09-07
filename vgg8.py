from torch import Tensor

def create_rpu_config(g_max=25.0):

    from aihwkit.simulator.configs import InferenceRPUConfig
    from aihwkit.simulator.configs.utils import BoundManagementType, WeightNoiseType
    #from aihwkit.simulator.noise_models import PCMLikeNoiseModel
    from aihwkit.inference import PCMLikeNoiseModel

    rpu_config = InferenceRPUConfig()
    rpu_config.backward.bound_management = BoundManagementType.NONE
    rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
    #rpu_config.forward.out_res = 1/256. # 8-bit ADC discretization.
    rpu_config.forward.out_res = -1.  # Turn off (output) ADC discretization.
    rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
    rpu_config.forward.w_noise = 0.02 # Some short-term w-noise.
    #rpu_config.forward.out_noise = 0.02 # Some output noise.

    rpu_config.noise_model = PCMLikeNoiseModel(g_max=g_max)

    return rpu_config

rpu_config0 = create_rpu_config(10)
rpu_config1 = create_rpu_config(20)
rpu_config2 = create_rpu_config(40)

print(rpu_config0)
print(rpu_config1)
print(rpu_config2)

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.parameter import Parameter, UninitializedParameter
#For finetuning
import torch.optim as optim
import time
import os
import copy

##For checking gradient function
from torch.autograd import Function
from torch.autograd import gradcheck

# Define relevant variables for the ML task
batch_size = 100
num_classes = 10
#learning_rate = 0.001
#num_epochs = 1
#device = torch.device('cpu')
device = torch.device('cuda')

#Loading the dataset and preprocessing
test_dataset = torchvision.datasets.CIFAR10(root = './data',
                                          train = False,
                                          transform = transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                          download=True)
validation_data = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

train_dataset = torchvision.datasets.CIFAR10(root = './data',
                                           train = True,
                                           transform = transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
                                           download = True)

train_data = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = batch_size,
                                           shuffle = True)

from torch.nn import Tanh, MaxPool2d, LogSoftmax, Flatten
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential

def multi_head(out0, out1, out2, prob):
  out = prob[0]*out0+prob[1]*out1 + prob[2]*out2
  return out

#Defining the convolutional neural network
class VGG8(nn.Module):
    def __init__(self, num_classes):
        super(VGG8, self).__init__()

        self.conv0_0 = AnalogConv2d(3, 48, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config0)
        self.conv0_1 = AnalogConv2d(3, 48, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config1)
        self.conv0_2 = AnalogConv2d(3, 48, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config2)

        self.relu0 = nn.ReLU()
        
        self.conv1_0 = AnalogConv2d(48, 48, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config0)
        self.conv1_1 = AnalogConv2d(48, 48, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config1)
        self.conv1_2 = AnalogConv2d(48, 48, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config2)

        self.batch1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, dilation=1)

        self.conv2_0 = AnalogConv2d(48, 96, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config0)
        self.conv2_1 = AnalogConv2d(48, 96, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config1)
        self.conv2_2 = AnalogConv2d(48, 96, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config2)

        self.relu2 = nn.ReLU()

        self.conv3_0 = AnalogConv2d(96, 96, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config0)
        self.conv3_1 = AnalogConv2d(96, 96, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config1)
        self.conv3_2 = AnalogConv2d(96, 96, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config2)

        self.batch3 = nn.BatchNorm2d(96)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, dilation=1)

        self.conv4_0 = AnalogConv2d(96, 144, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config0)
        self.conv4_1 = AnalogConv2d(96, 144, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config1)
        self.conv4_2 = AnalogConv2d(96, 144, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config2)

        self.relu4 = nn.ReLU()

        self.conv5_0 = AnalogConv2d(144, 144, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config0)
        self.conv5_1 = AnalogConv2d(144, 144, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config1)
        self.conv5_2 = AnalogConv2d(144, 144, kernel_size=3, stride=1, padding=1, rpu_config=rpu_config2)

        self.batch5 = nn.BatchNorm2d(144)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding=0, dilation=1)
        self.flatten5 = nn.Flatten()

        self.fc6_0 = AnalogLinear(2304, 384, rpu_config=rpu_config0)
        self.fc6_1 = AnalogLinear(2304, 384, rpu_config=rpu_config1)
        self.fc6_2 = AnalogLinear(2304, 384, rpu_config=rpu_config2)

        self.relu6 = nn.ReLU()

        self.fc7_0 = AnalogLinear(384, num_classes, rpu_config=rpu_config0)
        self.fc7_1 = AnalogLinear(384, num_classes, rpu_config=rpu_config1)
        self.fc7_2 = AnalogLinear(384, num_classes, rpu_config=rpu_config2)
        self.logsoft7 = nn.LogSoftmax()
        self.soft = nn.Softmax()


    def forward(self, x, alpha):
        out = multi_head(self.conv0_0(x), self.conv0_1(x), self.conv0_2(x), self.soft(alpha[0]))
        out = self.relu0(out)

        out = multi_head(self.conv1_0(out), self.conv1_1(out), self.conv1_2(out), self.soft(alpha[1]))
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.pool1(out)

        out = multi_head(self.conv2_0(out), self.conv2_1(out), self.conv2_2(out), self.soft(alpha[2]))
        out = self.relu2(out)

        out = multi_head(self.conv3_0(out), self.conv3_1(out), self.conv3_2(out), self.soft(alpha[3]))
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.pool3(out)

        out = multi_head(self.conv4_0(out), self.conv4_1(out), self.conv4_2(out), self.soft(alpha[4]))
        out = self.relu4(out)

        out = multi_head(self.conv5_0(out), self.conv5_1(out), self.conv5_2(out), self.soft(alpha[5]))
        out = self.batch5(out)
        out = self.relu5(out)
        out = self.pool5(out)
        out = self.flatten5(out)

        #out = out.reshape(out.size(0), -1)

        out = multi_head(self.fc6_0(out), self.fc6_1(out), self.fc6_2(out), self.soft(alpha[6]))

        out = self.relu6(out)
        out = multi_head(self.fc7_0(out), self.fc7_1(out), self.fc7_2(out), self.soft(alpha[7]))
        out = self.logsoft7(out)
        return out


alpha = torch.nn.Parameter(torch.ones(8,3)*0.33333, requires_grad=True)
#alpha = torch.nn.Parameter(torch.tensor([[ 1.6575,  0.1874, -0.8450],
#        [ 0.3373,  0.2998,  0.3629],
#        [ 0.9550,  0.5098, -0.4649],
#        [ 0.3179,  0.3560,  0.3261],
#        [ 0.8410,  0.8797, -0.7207],
#        [ 0.3303,  0.3359,  0.3338],
#        [ 0.3510,  0.3774,  0.2716],
#        [ 0.3426,  0.3178,  0.3396]]), requires_grad=True)
alpha.to(device)
print("Alpha", alpha)

model = VGG8(num_classes)
#model.load_state_dict(torch.load("./vgg8_cifar10.pt",map_location=torch.device('cpu')))
#model = torch.load("./vgg8_cifar10.pt",map_location=torch.device('cpu'))
#torch.load(model, 'vgg8_cifar10.pt')
model.cuda()

def test_step(validation_data, model, criterion):
    """Test trained network

    Args:
        validation_data (DataLoader): Validation set to perform the evaluation
        model (nn.Module): Trained model to be evaluated
        criterion (nn.CrossEntropyLoss): criterion to compute loss

    Returns: 
        test_dataset_loss: epoch loss of the train_dataset
        test_dataset_error: error of the test dataset
        test_dataset_accuracy: accuracy of the test dataset
    """
    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in validation_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images, alpha)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)

        _, predicted = torch.max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        test_dataset_accuracy = predicted_ok/total_images*100
        test_dataset_error = (1-predicted_ok/total_images)*100

    test_dataset_loss = total_loss / len(validation_data.dataset)

    return test_dataset_loss, test_dataset_error, test_dataset_accuracy

def noise_injection(model,t):
  model.eval()
  model.conv0_0.drift_analog_weights(t)
  model.conv0_1.drift_analog_weights(t)
  model.conv0_2.drift_analog_weights(t)
  model.conv1_0.drift_analog_weights(t)
  model.conv1_1.drift_analog_weights(t)
  model.conv1_2.drift_analog_weights(t)
  model.conv2_0.drift_analog_weights(t)
  model.conv2_1.drift_analog_weights(t)
  model.conv2_2.drift_analog_weights(t)
  model.conv3_0.drift_analog_weights(t)
  model.conv3_1.drift_analog_weights(t)
  model.conv3_2.drift_analog_weights(t)
  model.conv4_0.drift_analog_weights(t)
  model.conv4_1.drift_analog_weights(t)
  model.conv4_2.drift_analog_weights(t)
  model.conv5_0.drift_analog_weights(t)
  model.conv5_1.drift_analog_weights(t)
  model.conv5_2.drift_analog_weights(t)
  model.fc6_0.drift_analog_weights(t)
  model.fc6_1.drift_analog_weights(t)
  model.fc6_2.drift_analog_weights(t)
  model.fc7_0.drift_analog_weights(t)
  model.fc7_1.drift_analog_weights(t)
  model.fc7_2.drift_analog_weights(t)
  model.train()

def test_inference(model, criterion, test_data):
    
    from numpy import logspace, log10
    
    total_loss = 0
    predicted_ok = 0
    total_images = 0
    accuracy_pre = 0
    error_pre = 0
    
    # Create the t_inference_list using inference_time.
    # Generate the 9 values between 0 and the inference time using log10
    max_inference_time = 1e6
    n_times = 9
    t_inference_list = [0.0] + logspace(0, log10(float(max_inference_time)), n_times).tolist()

    # Simulation of inference pass at different times after training.
    for t_inference in t_inference_list:
        #model.drift_analog_weights(t_inference)
        noise_injection(model,t_inference)

        time_since = t_inference
        accuracy_post = 0
        error_post = 0
        predicted_ok = 0
        total_images = 0

        for images, labels in test_data:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            pred = model(images, alpha)
            loss = criterion(pred, labels)
            total_loss += loss.item() * images.size(0)

            _, predicted = torch.max(pred.data, 1)
            total_images += labels.size(0)
            predicted_ok += (predicted == labels).sum().item()
            accuracy_post = predicted_ok/total_images*100
            error_post = (1-predicted_ok/total_images)*100

        print(f'Error after inference: {error_post:.2f}\t'
              f'Accuracy after inference: {accuracy_post:.2f}%\t'
              f'Drift t={time_since: .2e}\t')


from torch.nn import CrossEntropyLoss
from aihwkit.optim import AnalogSGD

from torch import device, cuda

DEVICE = device('cuda' if cuda.is_available() else 'cpu')
print('Running the simulation on: ', DEVICE)

#model = LeNet5(num_classes)
from datetime import datetime
print(f'\n{datetime.now().time().replace(microsecond=0)} --- '
          f'Started Vgg8 Example')


criterion = CrossEntropyLoss()

optimizer = AnalogSGD(list(model.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)
MILESTONES = [50, 80, 100]
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2) #learning rate decay
optimizer.regroup_param_groups(model)


def train_step(train_data, model, criterion, optimizer):
  total_loss = 0

  model.train()

  for images, labels in train_data:
      images = images.to(DEVICE)
      #images.cuda()
      labels = labels.to(DEVICE)
      optimizer.zero_grad()

      # Add training Tensor to the model (input).
      #noise_injection(model,0.0)
      output = model(images, alpha)
      loss = criterion(output, labels)

      # Run training (backward propagation).
      loss.backward()

      # Optimize weights.
      optimizer.step()
      total_loss += loss.item() * images.size(0)
  train_dataset_loss = total_loss / len(train_data.dataset)
  return train_dataset_loss


train_losses = []
valid_losses = []
test_error = []
epochs = 100
print_every=1

# Train model
for epoch in range(0, epochs):
    # Train_step
    train_scheduler.step(epoch)
    train_loss = train_step(train_data, model, criterion, optimizer)
    train_losses.append(train_loss)

    if epoch % print_every == (print_every - 1):
        # Validate_step
        with torch.no_grad():
            valid_loss, error, accuracy = test_step(validation_data, model, criterion)
            valid_losses.append(valid_loss)
            test_error.append(error)

        print(f'Epoch: {epoch}\t'
              f'Train loss: {train_loss:.4f}\t'
              f'Valid loss: {valid_loss:.4f}\t'
              f'Test error: {error:.2f}%\t'
              f'Test accuracy: {accuracy:.2f}%\t')

print("Alpha", alpha)
m = nn.Softmax(dim=1)
output = m(alpha)

print("Probability", output)

print("Inference for t_time")
test_inference(model, criterion, validation_data)

print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'Completed Vgg8 Example')

torch.save(model.state_dict(), 'vgg8_cifar10_2.pt')

alpha_main = alpha.clone().detach()
model_main = VGG8(num_classes)
model_main.load_state_dict(model.state_dict())

def train_step_inject(train_data, model, criterion, optimizer):
  total_loss = 0

  model.train()

  for images, labels in train_data:
      images = images.to(DEVICE)
      #images.cuda()
      labels = labels.to(DEVICE)
      optimizer.zero_grad()

      # Add training Tensor to the model (input).
      noise_injection(model,0.0)
      output = model(images, alpha)
      loss = criterion(output, labels)

      # Run training (backward propagation).
      loss.backward()

      # Optimize weights.
      optimizer.step()
      total_loss += loss.item() * images.size(0)
  train_dataset_loss = total_loss / len(train_data.dataset)
  return train_dataset_loss


train_losses = []
valid_losses = []
test_error = []
epochs = 10
print_every=1

# Train model
for epoch in range(0, epochs):
    # Train_step
    train_loss = train_step_inject(train_data, model, criterion, optimizer)
    train_losses.append(train_loss)

    if epoch % print_every == (print_every - 1):
        # Validate_step
        with torch.no_grad():
            valid_loss, error, accuracy = test_step(validation_data, model, criterion)
            valid_losses.append(valid_loss)
            test_error.append(error)

        print(f'Epoch: {epoch}\t'
              f'Train loss: {train_loss:.4f}\t'
              f'Valid loss: {valid_loss:.4f}\t'
              f'Test error: {error:.2f}%\t'
              f'Test accuracy: {accuracy:.2f}%\t')

print("Alpha", alpha)
m = nn.Softmax(dim=1)
output = m(alpha)

print("Probability", output)

print("Inference for t_time")
test_inference(model, criterion, validation_data)

print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'Completed Vgg8 Example')

torch.save(model.state_dict(), 'vgg8_cifar10_inject.pt')

factors = torch.FloatTensor([[1,1.286,1.858],[1,1.277817,1.833343],[1,1.263823,1.791472],[1,1.277817,1.833343],[1,1.263823,1.791472],[1,1.20034,1.601032],[1,1.260201,1.780606]])
min_factors = torch.FloatTensor([[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1]])

def modified_sqrtmean(f_e, f_a, alpha):
  Soft = nn.Softmax(dim=1)
  p = Soft(alpha)
  return torch.sqrt(torch.mean(p*torch.square(f_e-f_a)))


RSME = modified_sqrtmean(min_factors, factors, alpha)

print("RSME", RSME)

### Clone alpha
alpha = alpha_main.clone().detach()
alpha.requires_grad_(True)
#alpha = torch.nn.Parameter(alpha_main, requires_grad=True)
print("Alpha", alpha)

### Clone model
model = VGG8(num_classes)
model.load_state_dict(model_main.state_dict())


def train_step_inject2(train_data, model, criterion, optimizer):
  total_loss = 0

  model.train()

  for images, labels in train_data:
      images = images.to(DEVICE)
      #images.cuda()
      labels = labels.to(DEVICE)
      optimizer.zero_grad()

      # Add training Tensor to the model (input).
      noise_injection(model,0.0)
      output = model(images, alpha)
      loss = 0.9*criterion(output, labels) + 0.1*modified_sqrtmean(min_factors, factors, alpha)

      # Run training (backward propagation).
      loss.backward()

      # Optimize weights.
      optimizer.step()
      total_loss += loss.item() * images.size(0)
  train_dataset_loss = total_loss / len(train_data.dataset)
  return train_dataset_loss


train_losses = []
valid_losses = []
test_error = []
epochs = 10
print_every=1

# Train model
for epoch in range(0, epochs):
    # Train_step
    train_loss = train_step_inject2(train_data, model, criterion, optimizer)
    train_losses.append(train_loss)

    if epoch % print_every == (print_every - 1):
        # Validate_step
        with torch.no_grad():
            valid_loss, error, accuracy = test_step(validation_data, model, criterion)
            valid_losses.append(valid_loss)
            test_error.append(error)

        print(f'Epoch: {epoch}\t'
              f'Train loss: {train_loss:.4f}\t'
              f'Valid loss: {valid_loss:.4f}\t'
              f'Test error: {error:.2f}%\t'
              f'Test accuracy: {accuracy:.2f}%\t')

print("Alpha", alpha)
m = nn.Softmax(dim=1)
output = m(alpha)

print("Probability", output)

print("Inference for t_time")
test_inference(model, criterion, validation_data)

print(f'{datetime.now().time().replace(microsecond=0)} --- '
          f'Completed Vgg8 Example')

torch.save(model.state_dict(), 'vgg8_cifar10_inject2.pt')
