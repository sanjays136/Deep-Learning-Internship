

import itertools
import math
from sklearn.utils import shuffle
import wandb

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import copy
import argparse
#!pip install wandb
import wandb
import socket
socket.setdefaulttimeout(30)
wandb.login()


def normalize_data(x):
  x_norm = x.astype('float32')
  x_norm = x_norm / 255.0
  return x_norm

def func(activation,a_k1):
  a_k = np.clip(a_k1, -55, 55)
  if(activation == "tanh"):
    a_k = np.tanh(a_k)
  elif(activation == "sigmoid"):
    a_k = 1/(1 + np.exp(-1*a_k))
  else:
    a_k = np.maximum(0,a_k)
  return a_k

def derivativeFun(activation,a_k1):
  a_k = np.clip(a_k1, -55, 55)
  activationResult = func(activation,a_k)
  if(activation == "tanh"):
    activationResult = 1 - (activationResult**2)
  elif(activation == "sigmoid"):
    activationResult = activationResult - (activationResult**2)
  else:
    activationResult = np.where(a_k > 0, 1, 0)
  return activationResult

def decision(a_k):
  a_k = np.exp(a_k - np.max(a_k))
  a_k = a_k / sum(a_k)
  return a_k

def forwardProp(inputX,activation,weights,bias):
  h_k = inputX
  PreActivations = list()
  PostActivations = list()
  PostActivations.append(h_k)
  layers = len(weights) - 1
  for k in range(0,layers):
    a_k = bias[k] + np.dot(weights[k],h_k)
    PreActivations.append(a_k)
    h_k = func(activation,a_k)
    PostActivations.append(h_k)
  a_k = bias[layers] + np.matmul(weights[layers],h_k)
  PreActivations.append(a_k)
  yPred = decision(a_k)
  return PreActivations,PostActivations,yPred

def backProp(real, pred, h_k, weights, activation, PreActivations,lossFunction):
    a_l_L_theta = pred - real
    if lossFunction == "cross_entropy":
      a_l_L_theta = pred - real
    elif lossFunction == "mean_squared_error":
      a_l_L_theta = np.multiply(np.multiply((pred - real), pred), (1 - pred))

    currentActivationGradient = a_l_L_theta
    WeightGradients = []
    biasGradients = []
    layers = len(weights) - 1

    for i in range(layers, -1, -1):
        W_i_L_theta = currentActivationGradient*np.transpose(h_k[i])
        WeightGradients.insert(0, W_i_L_theta)
        b_i_L_theta = np.sum(currentActivationGradient, axis=0, keepdims=True)
        biasGradients.insert(0, b_i_L_theta)

        if i > 0:
            h_i_prev_L_theta = np.matmul(weights[i].T, currentActivationGradient)
            currentActivationGradient = h_i_prev_L_theta * derivativeFun(activation, PreActivations[i - 1])

    return WeightGradients, biasGradients

def randomizer(dim1,dim2,init_weight):
  std_dev = 0.1
  if(init_weight == "Xavier"):
    variance = 2.0 / (dim1 + dim2)
    std_dev = np.sqrt(variance)
  return std_dev

def stochastic_gradient_descent(nodesPerLayer, x_flatten_train, y_encoded, batch_size,activationFunc,epochs,lr,x_flatten_test,y_flatten_test,x_val,y_val,init_weight,lambda_reg,lossFunction):
    weights = [np.random.randn(nodesPerLayer[i], nodesPerLayer[i-1]) * randomizer(nodesPerLayer[i], nodesPerLayer[i-1],init_weight) for i in range(1, len(nodesPerLayer))]
    bias = [np.random.randn(nodesPerLayer[i], 1) * 0.1 for i in range(1, len(nodesPerLayer))]

    num_batches = len(x_flatten_train)

    for epoch in range(epochs):
        for batch in range(0,num_batches):
            if(batch % (num_batches/10) == 0):
              print("batch: " + str(batch))
            start = batch * batch_size
            end = (batch + 1) * batch_size

            batch_x = x_flatten_train[start:end]
            batch_y = y_encoded[start:end]

            batch_Wdelta = [np.zeros_like(w) for w in weights]
            batch_Bdelta = [np.zeros_like(b) for b in bias]

            for j in range(len(batch_x)):
                A, B, C = forwardProp(batch_x[j], activationFunc,weights, bias)
                Wdelta, Bdelta = backProp(batch_y[j], C, B, weights, activationFunc, A,lossFunction)

                for k in range(len(batch_Wdelta)):
                    batch_Wdelta[k] += Wdelta[k]
                    batch_Bdelta[k] += Bdelta[k]

            for k in range(len(weights)):
                weights[k] = ((1 - (lr * lambda_reg / batch_size)) * weights[k]) - lr * (batch_Wdelta[k] / batch_size)
                bias[k] -= lr * (batch_Bdelta[k] / batch_size)
        accuracy,loss = testModel(weights,bias,x_flatten_test,y_flatten_test,activationFunc,lossFunction)
        Valaccuracy,Valloss = testModel(weights,bias,x_val,y_val,activationFunc,lossFunction)
        #print({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})
        wandb.log({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})

    return weights, bias

def momentum_gradient_descent(nodesPerLayer, x_flatten_train, y_encoded, gamma, batch_size,activationFunc,epochs,lr,x_flatten_test,y_flatten_test,x_val,y_val,init_weight,lambda_reg,lossFunction):

    weights = [np.random.randn(nodesPerLayer[i], nodesPerLayer[i-1]) * randomizer(nodesPerLayer[i], nodesPerLayer[i-1],init_weight) for i in range(1, len(nodesPerLayer))]
    bias = [np.random.randn(nodesPerLayer[i], 1) * 0.1 for i in range(1, len(nodesPerLayer))]

    Wdelta = [np.zeros((nodesPerLayer[i], nodesPerLayer[i-1])) for i in range(1, len(nodesPerLayer))]
    Bdelta = [np.zeros((nodesPerLayer[i], 1)) for i in range(1, len(nodesPerLayer))]

    num_batches = len(x_flatten_train)

    for epoch in range(epochs):
        for batch in range(0,num_batches):
            if(batch % (num_batches/10) == 0):
              print("batch :" + str(batch))
            start = batch * batch_size
            end = (batch + 1) * batch_size

            batch_x = x_flatten_train[start:end]
            batch_y = y_encoded[start:end]

            batch_Wdelta = [np.zeros_like(w) for w in weights]
            batch_Bdelta = [np.zeros_like(b) for b in bias]

            for j in range(len(batch_x)):
                A, B, C = forwardProp(batch_x[j], activationFunc, weights, bias)
                CurrWdelta, CurrBdelta = backProp(batch_y[j], C, B, weights,activationFunc, A,lossFunction)

                for k in range(len(batch_Wdelta)):
                    batch_Wdelta[k] += CurrWdelta[k]
                    batch_Bdelta[k] += CurrBdelta[k]

            for k in range(len(weights)):
                Wdelta[k] = gamma * Wdelta[k] + lr * batch_Wdelta[k] / batch_size
                Bdelta[k] = gamma * Bdelta[k] + lr * batch_Bdelta[k] / batch_size

                weights[k] = ((1 - (lr * lambda_reg / batch_size)) * weights[k]) -Wdelta[k]
                bias[k] -= Bdelta[k]
        accuracy,loss = testModel(weights,bias,x_flatten_test,y_flatten_test,activationFunc,lossFunction)
        Valaccuracy,Valloss = testModel(weights,bias,x_val,y_val,activationFunc,lossFunction)
        #print({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})
        wandb.log({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})

    return weights, bias

def nesterov_gradient_descent(nodesPerLayer,x_flatten_train,y_encoded,gamma, batch_size,activationFunc,epochs,lr,x_flatten_test,y_flatten_test,x_val,y_val,init_weight,lambda_reg,lossFunction):
    weights = [np.random.randn(nodesPerLayer[i], nodesPerLayer[i-1]) * randomizer(nodesPerLayer[i], nodesPerLayer[i-1],init_weight) for i in range(1, len(nodesPerLayer))]
    bias = [np.random.randn(nodesPerLayer[i], 1) * 0.1 for i in range(1, len(nodesPerLayer))]

    num_batches = len(x_flatten_train)

    for epoch in range(epochs):
        for batch in range(0,num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            batch_x = x_flatten_train[start:end]
            batch_y = y_encoded[start:end]

            lookahead_weights = [w - gamma * dw for w, dw in zip(weights, weights)]
            lookahead_bias = [b - gamma * db for b, db in zip(bias, bias)]

            for j in range(len(batch_x)):
                A, B, C = forwardProp(batch_x[j], activationFunc, lookahead_weights, lookahead_bias)
                CurrWdelta, CurrBdelta = backProp(batch_y[j], C, B, lookahead_weights, activationFunc, A,lossFunction)

                for k in range(len(weights)):
                    weights[k] = ((1 - (lr * lambda_reg / batch_size)) * weights[k]) - lr * CurrWdelta[k]
                    bias[k] -= lr * CurrBdelta[k]
        accuracy,loss = testModel(weights,bias,x_flatten_test,y_flatten_test,activationFunc,lossFunction)
        Valaccuracy,Valloss = testModel(weights,bias,x_val,y_val,activationFunc,lossFunction)
        #print({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})
        wandb.log({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})
    return weights, bias

def rmsprop(nodesPerLayer, x_flatten_train, y_encoded, beta, eps, epochs, batch_size, lr,activationFunc,x_flatten_test,y_flatten_test,x_val,y_val,init_weight,lambda_reg,lossFunction):
    weights = [np.random.randn(nodesPerLayer[i], nodesPerLayer[i-1]) * randomizer(nodesPerLayer[i], nodesPerLayer[i-1],init_weight) for i in range(1, len(nodesPerLayer))]
    bias = [np.random.randn(nodesPerLayer[i], 1) * 0.1 for i in range(1, len(nodesPerLayer))]

    rmsweights = [np.zeros((nodesPerLayer[i], nodesPerLayer[i-1])) for i in range(1, len(nodesPerLayer))]
    rmsbias = [np.zeros((nodesPerLayer[i], 1)) for i in range(1, len(nodesPerLayer))]

    num_batches = len(x_flatten_train)

    for epoch in range(epochs):
        for batch in range(0,num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            batch_x = x_flatten_train[start:end]
            batch_y = y_encoded[start:end]

            batch_w_delta = [np.zeros_like(w) for w in weights]
            batch_b_delta = [np.zeros_like(b) for b in bias]

            for j in range(len(batch_x)):
                A, B, C = forwardProp(batch_x[j], activationFunc,  weights, bias)
                CurrWdelta, CurrBdelta = backProp(batch_y[j], C, B, weights, activationFunc, A,lossFunction)

                for k in range(len(CurrWdelta)):
                    batch_w_delta[k] += CurrWdelta[k]
                    batch_b_delta[k] += CurrBdelta[k]

            for k in range(len(batch_w_delta)):
                rmsweights[k] = beta * rmsweights[k] + (1 - beta) * (batch_w_delta[k] ** 2)
                rmsbias[k] = beta * rmsbias[k] + (1 - beta) * (batch_b_delta[k] ** 2)

                weights[k] =((1 - (lr * lambda_reg / batch_size)) * weights[k]) -( (lr * batch_w_delta[k]) / (np.sqrt(rmsweights[k]) + eps))
                bias[k] -= (lr * batch_b_delta[k]) / (np.sqrt(rmsbias[k]) + eps)
        accuracy,loss = testModel(weights,bias,x_flatten_test,y_flatten_test,activationFunc,lossFunction)
        Valaccuracy,Valloss = testModel(weights,bias,x_val,y_val,activationFunc,lossFunction)
        #print({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})
        wandb.log({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})

    return weights, bias

def adam(nodesPerLayer, x_flatten_train, y_encoded, beta1, beta2, eps, batch_size, lr,activationFunc,epochs,x_flatten_test,y_flatten_test,x_val,y_val,init_weight,lambda_reg,lossFunction):
    # Initialize weights and biases
    weights = [np.random.randn(nodesPerLayer[i], nodesPerLayer[i-1]) * randomizer(nodesPerLayer[i], nodesPerLayer[i-1],init_weight) for i in range(1, len(nodesPerLayer))]
    bias = [np.random.randn(nodesPerLayer[i], 1) * 0.1 for i in range(1, len(nodesPerLayer))]

    # Initialize Adam parameters
    m_weights = [np.zeros((nodesPerLayer[i], nodesPerLayer[i-1])) for i in range(1, len(nodesPerLayer))]
    v_weights = [np.zeros((nodesPerLayer[i], nodesPerLayer[i-1])) for i in range(1, len(nodesPerLayer))]
    m_bias = [np.zeros((nodesPerLayer[i], 1)) for i in range(1, len(nodesPerLayer))]
    v_bias = [np.zeros((nodesPerLayer[i], 1)) for i in range(1, len(nodesPerLayer))]

    num_batches = len(x_flatten_train)

    for epoch in range(epochs):
        for batch in range(0,num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size

            batch_x = x_flatten_train[start:end]
            batch_y = y_encoded[start:end]

            batch_w_delta = [np.zeros_like(w) for w in weights]
            batch_b_delta = [np.zeros_like(b) for b in bias]

            for j in range(len(batch_x)):
                A, B, C = forwardProp(batch_x[j], activationFunc,  weights, bias)
                CurrWdelta, CurrBdelta = backProp(batch_y[j], C, B, weights, activationFunc, A,lossFunction)

                for k in range(len(CurrWdelta)):
                    batch_w_delta[k] += CurrWdelta[k]
                    batch_b_delta[k] += CurrBdelta[k]

            for k in range(len(batch_w_delta)):
                m_weights[k] = beta1 * m_weights[k] + (1 - beta1) * batch_w_delta[k]
                v_weights[k] = beta2 * v_weights[k] + (1 - beta2) * (batch_w_delta[k] ** 2)
                m_bias[k] = beta1 * m_bias[k] + (1 - beta1) * batch_b_delta[k]
                v_bias[k] = beta2 * v_bias[k] + (1 - beta2) * (batch_b_delta[k] ** 2)

                m_weights_hat = m_weights[k] / (1 - beta1 ** (epoch + 1))
                v_weights_hat = v_weights[k] / (1 - beta2 ** (epoch + 1))
                m_bias_hat = m_bias[k] / (1 - beta1 ** (epoch + 1))
                v_bias_hat = v_bias[k] / (1 - beta2 ** (epoch + 1))

                weights[k] = ((1 - (lr * lambda_reg / batch_size)) * weights[k]) - ( (lr * m_weights_hat) / (np.sqrt(v_weights_hat) + eps))
                bias[k] -= (lr * m_bias_hat) / (np.sqrt(v_bias_hat) + eps)

        accuracy,loss = testModel(weights,bias,x_flatten_test,y_flatten_test,activationFunc,lossFunction)
        Valaccuracy,Valloss = testModel(weights,bias,x_val,y_val,activationFunc,lossFunction)
        #print({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})
        wandb.log({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})


    return weights, bias

def nadam(nodesPerLayer, x_flatten_train, y_encoded, beta1, beta2, eps, batch_size, lr,activationFunc,epochs,x_flatten_test,y_flatten_test,x_val,y_val,init_weight,lambda_reg,lossFunction):
    # Initialize weights and biases
    weights = [np.random.randn(nodesPerLayer[i], nodesPerLayer[i-1]) * randomizer(nodesPerLayer[i], nodesPerLayer[i-1],init_weight) for i in range(1, len(nodesPerLayer))]
    bias = [np.random.randn(nodesPerLayer[i], 1) * 0.1 for i in range(1, len(nodesPerLayer))]

    # Initialize Nadam parameters
    m_weights = [np.zeros((nodesPerLayer[i], nodesPerLayer[i-1])) for i in range(1, len(nodesPerLayer))]
    v_weights = [np.zeros((nodesPerLayer[i], nodesPerLayer[i-1])) for i in range(1, len(nodesPerLayer))]
    m_bias = [np.zeros((nodesPerLayer[i], 1)) for i in range(1, len(nodesPerLayer))]
    v_bias = [np.zeros((nodesPerLayer[i], 1)) for i in range(1, len(nodesPerLayer))]

    num_batches = len(x_flatten_train) // batch_size

    for epoch in range(epochs):
        for batch in range(0,num_batches):
            start = batch * batch_size
            end = (batch + 1) * batch_size
            batch_x = x_flatten_train[start:end]
            batch_y = y_encoded[start:end]

            batch_w_delta = [np.zeros_like(w) for w in weights]
            batch_b_delta = [np.zeros_like(b) for b in bias]

            for j in range(len(batch_x)):
                A, B, C = forwardProp(batch_x[j], activationFunc, weights, bias)
                CurrWdelta, CurrBdelta = backProp(batch_y[j], C, B, weights, activationFunc, A,lossFunction)

                for k in range(len(CurrWdelta)):
                    batch_w_delta[k] += CurrWdelta[k]
                    batch_b_delta[k] += CurrBdelta[k]

            for k in range(len(batch_w_delta)):
                m_weights[k] = beta1 * m_weights[k] + (1 - beta1) * batch_w_delta[k]
                v_weights[k] = beta2 * v_weights[k] + (1 - beta2) * (batch_w_delta[k] ** 2)
                m_bias[k] = beta1 * m_bias[k] + (1 - beta1) * batch_b_delta[k]
                v_bias[k] = beta2 * v_bias[k] + (1 - beta2) * (batch_b_delta[k] ** 2)

                m_weights_hat = m_weights[k] / (1 - beta1 ** (epoch + 1))
                v_weights_hat = v_weights[k] / (1 - beta2 ** (epoch + 1))
                m_bias_hat = m_bias[k] / (1 - beta1 ** (epoch + 1))
                v_bias_hat = v_bias[k] / (1 - beta2 ** (epoch + 1))

                weights[k] =((1 - (lr * lambda_reg / batch_size)) * weights[k]) - ( lr * (beta1 * m_weights_hat + (1 - beta1) * batch_w_delta[k]) / (np.sqrt(v_weights_hat) + eps))
                bias[k] -= lr * (beta1 * m_bias_hat + (1 - beta1) * batch_b_delta[k]) / (np.sqrt(v_bias_hat) + eps)
        accuracy,loss = testModel(weights,bias,x_flatten_test,y_flatten_test,activationFunc,lossFunction)
        Valaccuracy,Valloss = testModel(weights,bias,x_val,y_val,activationFunc,lossFunction)
        wandb.log({"val_loss":Valloss,"val_accuracy":Valaccuracy,"loss":loss,"accuracy":accuracy,"epoch":epoch})

    return weights, bias

def gradient_descent(nodesPerLayer,x_flatten_train,y_encoded,activationFunc,epochs,lr):
  weights = list()
  bias = list()
  w = np.random.randn(nodesPerLayer[i],nodesPerLayer[i-1])*0.1
  b =  np.random.randn(nodesPerLayer[i],1)
  weights.append(w)
  bias.append(b)
  Wdelta = list()
  Bdelta = list()
  for i in range(0,epochs):
    Wdelta.clear()
    Bdelta.clear()
    for j in range(0,len(y_encoded)):
      A,B,C = forwardProp(x_flatten_train[j],activationFunc,weights,bias)
      CurrWdelta,CurrBdelta = backProp(y_encoded[j],C,B,weights,activationFunc,A)
      if( len(Wdelta) == 0):
        Wdelta =  copy.deepcopy(CurrWdelta)
        Bdelta = copy.deepcopy(CurrBdelta)
      else:
        for k in range(0,len(Wdelta)):
          Wdelta[k] = Wdelta[k] + CurrWdelta[k]
          Bdelta[k] = Bdelta[k] + CurrBdelta[k]
      if(j%1000 == 0):
        print(j/1000)
    for k in range(0,len(weights)):
      weights[k] = weights[k] - lr*Wdelta[k]
      bias[k] = bias[k] - lr*Bdelta[k]
  return weights,bias

def executeTraining(config,x_train,y_train,x_flatten_test,y_flatten_test,x_val,y_val):
  FinalWeights = list()
  FinalBias = list()
  beta1 = config.beta1
  beta2 = config.beta2
  eps = config.epsilon
  batch_size = config.batch_size
  layers = config.num_layers
  lr = config.learning_rate
  epochs = config.epochs
  activationFunc = config.activation
  optimizer = config.optimizer
  weightInit = config.weight_init
  lambda_reg = config.weight_decay
  nodesPerLayer = list()
  nodesPerLayer.append(784)
  for i in range(0,layers):
    nodesPerLayer.append(config.hidden_size)
  nodesPerLayer.append(10)

  lossFunction = config.loss
  gamma = config.momentum
  betarms = config.beta
  if(optimizer == "gradient_descent"):
    FinalWeights, FinalBias = gradient_descent(nodesPerLayer,x_train,y_train,activationFunc,epochs,lr)
  elif(optimizer == "sgd"):
    FinalWeights, FinalBias = stochastic_gradient_descent(nodesPerLayer,x_train,y_train,batch_size,activationFunc,epochs,lr,x_flatten_test,y_flatten_test,x_val,y_val,weightInit,lambda_reg,lossFunction)
  elif(optimizer == "momentum"):
    FinalWeights, FinalBias = momentum_gradient_descent(nodesPerLayer,x_train,y_train,gamma,batch_size,activationFunc,epochs,lr,x_flatten_test,y_flatten_test,x_val,y_val,weightInit,lambda_reg,lossFunction)
  elif(optimizer == "nag"):
    FinalWeights, FinalBias = nesterov_gradient_descent(nodesPerLayer,x_train,y_train,gamma,batch_size,activationFunc,epochs,lr,x_flatten_test,y_flatten_test,x_val,y_val,weightInit,lambda_reg,lossFunction)
  elif(optimizer == "rmsprop"):
    FinalWeights, FinalBias =rmsprop(nodesPerLayer,x_train,y_train,betarms,eps,epochs,batch_size,lr,activationFunc,x_flatten_test,y_flatten_test,x_val,y_val,weightInit,lambda_reg,lossFunction)
  elif(optimizer == "adam"):
    FinalWeights, FinalBias = adam(nodesPerLayer, x_train, y_train, beta1, beta2, eps, batch_size,lr,activationFunc,epochs,x_flatten_test,y_flatten_test,x_val,y_val,weightInit,lambda_reg,lossFunction)
  elif(optimizer == "nadam"):
    FinalWeights, FinalBias = nadam(nodesPerLayer, x_train, y_train, beta1, beta2, eps, batch_size,lr,activationFunc,epochs,x_flatten_test,y_flatten_test,x_val,y_val,weightInit,lambda_reg,lossFunction)
  return FinalWeights,FinalBias

def testModel(weights,bias,x_test,y_test,activationFun,lossFunction):
  count = 0
  loss = 0.0

  for i in range(0,x_test.shape[0]):
    A,B,C = forwardProp(x_test[i],activationFun,weights,bias)
    if( y_test[i]==np.argmax(C)):
      count+=1

    if(lossFunction == "mean_squared_error"):
      loss += (np.argmax(C) -  y_test[i])**2
    else:
      loss += -np.log(C)[y_test[i]][0]

  loss /= y_test.shape[0]
  acc = (count/y_test.shape[0])
  return acc,loss

def import_data(dataset):
  if(dataset == "mnist"):
   return mnist.load_data()
  else:
   return fashion_mnist.load_data()

def load_data(dataset):

  (x_train, y_train), (x_test, y_test) = import_data(dataset)
  x_train, y_train = shuffle(x_train, y_train)
  val_size = int(x_train.shape[0] * 0.1)

  y_val = y_train[:val_size]
  y_new_train = y_train[val_size:]

  x_flatten_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2],1)
  x_flatten_train = normalize_data(x_flatten_train)

  y_encoded = np.zeros((y_new_train.shape[0], 10))
  y_encoded[np.arange(y_new_train.shape[0]), y_new_train] = 1
  y_new_train = y_encoded.reshape(y_new_train.shape[0],10,1)


  x_flatten_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2],1)
  x_flatten_test = normalize_data(x_flatten_test)


  x_val = x_flatten_train[:val_size]


  x_new_train = x_flatten_train[val_size:]


  return x_new_train, y_new_train,x_val,y_val,x_flatten_test,y_test

def plot_diffClasses():
  (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
  names = ["T-shirt","Trouser","Pullover shirt","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
  variousSamples = list()
  classes = set(y_train)
  for i in classes:
    ind = np.where(y_train == i)[0][0]
    variousSamples.append(wandb.Image(x_train[ind],caption = names[i]))

  wandb.log({"examples": variousSamples})

def GenConfusion(weights,bias,x_test,y_test,activationFun):
  prediction = list()
  names = ["T-shirt","Trouser","Pullover shirt","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
  for i in range(0,x_test.shape[0]):
    A,B,C = forwardProp(x_test[i],activationFun,"crossEntropy",weights,bias)
    prediction.append(np.argmax(C))
  wandb.log({"conf_mat" : wandb.plot.confusion_matrix(y_true=y_test, preds=prediction,class_names=names)})
  return

def fitModel(args):
  
  wandb.run.name = "hidden_" + str(args.num_layers)+"_batchSize_"+str(args.batch_size)+"_acc_"+ args.activation

  x_new_train, y_new_train,x_val,y_val,x_flatten_test,y_flatten_test = load_data(args.dataset)
  trainedWeights,trainedBias = executeTraining(args,x_new_train,y_new_train,x_flatten_test,y_flatten_test,x_val,y_val)


def parse_arguments():
  parser = argparse.ArgumentParser(description='Training Parameters')
  parser.add_argument('-wp', '--wandb_project', type=str, default='AssignmentDL_1',
                        help='Project name')
  
  parser.add_argument('-we', '--wandb_entity', type=str, default='Entity_DL',
                        help='Wandb Entity')
  
  parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist',choices=["mnist", "fashion_mnist"],
                        help='Dataset choice: fashion_mnist , mnist')
  
  parser.add_argument('-e', '--epochs', type=int, default=10,help='Number of epochs for training network')

  parser.add_argument('-b', '--batch_size', type=int, default=64,help='Batch size for training neural network')

  parser.add_argument('-l', '--loss', type=str, default='cross_entropy',choices=["cross_entropy", "mean_squared_error"],help='Choice of mean_squared_error or cross_entropy')
  
  parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices = ["sgd", "momentum", "ngd", "rmsprop", "adam", "nadam"],help='Choice of optimizer')
   
  parser.add_argument('-lr', '--learning_rate', type=int, default=0.001, help='Learning rate')

  parser.add_argument( '-m', '--momentum', type=int, default=0.65, help='Momentum parameter')

  parser.add_argument('-beta', '--beta', type=int, default=0.58, help='Beta parameter')

  parser.add_argument('-beta1', '--beta1', type=int, default=0.89, help='Beta1 parameter')

  parser.add_argument('-beta2', '--beta2', type=int, default=0.85, help='Beta2 parameter')

  parser.add_argument( '-eps', '--epsilon', type=int, default=0.000001, help='Epsilon used by optimizers')

  parser.add_argument( '-w_i', '--weight_init',type=str, default="Xavier",choices=["random", "Xavier"], help='randomizer for weights')

  parser.add_argument('-w_d','--weight_decay',  type=int, default=0.0005, help='Weight decay parameter')

  parser.add_argument( '-nhl', '--num_layers',type=int, default=3, help='Number of hidden layers')
  
  parser.add_argument( '-sz','--hidden_size', type=int, default=128, help='Number of neurons in each layer')

  parser.add_argument( '-a','--activation', type=str, default="relu",choices=["sigmoid", "tanh", "relu"], help='activation functions')

  return parser.parse_args()

args = parse_arguments()

wandb.init(project=args.wandb_project)
fitModel(args)
 