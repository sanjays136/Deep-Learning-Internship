from keras.datasets import fashion_mnist
import numpy as np
import wandb
import plotly.graph_objs as go
from sklearn.metrics import confusion_matrix

def draw_confusion_matrix(y_pred, y_true) :
  y_pred = np.argmax(y_pred, axis=1)
  classes = []
  classes.append("T-shirt/top")
  classes.extend(['Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
  conf_matrix = confusion_matrix(y_pred, y_true, labels=range(len(classes)))
  #getting diagonal elements
  conf_diagonal_matrix = np.eye(len(conf_matrix)) * conf_matrix
  np.fill_diagonal(conf_matrix, 0)
  conf_matrix = conf_matrix.astype('float')
  n_confused = np.sum(conf_matrix)
  conf_matrix[conf_matrix == 0] = np.nan
  #giving red shades to non diagonal elements
  conf_matrix = go.Heatmap({'coloraxis': 'coloraxis1', 'x': classes, 'y': classes, 'z': conf_matrix, 'hoverongaps':False, 'hovertemplate': 'Predicted %{y}<br>Instead of %{x}<br>On %{z} examples<extra></extra>'})
  conf_diagonal_matrix = conf_diagonal_matrix.astype('float')
  n_right = np.sum(conf_diagonal_matrix)
  conf_diagonal_matrix[conf_diagonal_matrix == 0] = np.nan
  #giving green shade to diagonal elements
  conf_diagonal_matrix = go.Heatmap({'coloraxis': 'coloraxis2', 'x': classes, 'y': classes, 'z': conf_diagonal_matrix,'hoverongaps':False, 'hovertemplate': 'Predicted %{y} just right<br>On %{z} examples<extra></extra>'})
  fig = go.Figure((conf_diagonal_matrix, conf_matrix))
  transparent = 'rgba(0, 0, 0, 0)'
  n_total = n_right + n_confused
  fig.update_layout({'coloraxis1': {'colorscale': [[0, transparent], [0, 'rgba(180, 0, 0, 0.05)'], [1, f'rgba(180, 0, 0, {max(0.2, (n_confused/n_total) ** 0.5)})']], 'showscale': False}})
  fig.update_layout({'coloraxis2': {'colorscale': [[0, transparent], [0, f'rgba(0, 180, 0, {min(0.8, (n_right/n_total) ** 2)})'], [1, 'rgba(0, 180, 0, 1)']], 'showscale': False}})
  xaxis = {'title':{'text':'y_true'}, 'showticklabels':False}
  yaxis = {'title':{'text':'y_pred'}, 'showticklabels':False}
  fig.update_layout(title={'text':'Confusion matrix', 'x':0.5}, paper_bgcolor=transparent, plot_bgcolor=transparent, xaxis=xaxis, yaxis=yaxis)
  wandb.log({'Heat Map Confusion Matrix': wandb.data_types.Plotly(fig)})
  return 0




def plot_sample_images(X_train, Y_train):
    classes = []
    classes.append("T-shirt/top")
    classes.extend(['Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'])
    Images = []
    visited = set()
    for i in range(Y_train.shape[0]):
        class_num = Y_train[i]
        if(not(class_num in visited)):
            Images.append(wandb.Image(X_train[i], caption=classes[class_num]))
            visited.add(class_num)
        if(len(visited) == 10):
            break
    wandb.log({"Examples for each class": Images})
    return classes



#dimensions : 
#X_input -> n
#a_i,h_i : n x (ith hidden_layer_size)
#W_list[i] : (ith hidden_layer_size) x ({i-1}th hidden_layer_size)
#b_list[i] : (ith hidden_layer_size) x 1
#Output : h_list (h_1,h_2....h_{num_hidden_layers}), a_list (a_1,a_2....a_{num_hidden_layers+1})), y

def feed_forward_neural_network(X,W_list,b_list,activation_function, num_hidden_layers, hidden_layer_size):
  X.reshape(X.shape[0],784)
  size = X.shape[0]
  a_list = []
  h_list = []
  y = []
  h_prev = X
  for i in range(0,num_hidden_layers):
    a_i = b_list[i].flatten()+np.dot(h_prev,W_list[i].T)
    a_list.append(a_i)
    if activation_function == 'sigmoid' :
      a_i = np.where(a_i<-15,-15,a_i)
      a_i = np.where(a_i>15,15,a_i)
      h_prev = 1 /( 1 + np.exp(-1 * a_i))
    elif activation_function == 'tanh' :
      h_prev = np.tanh(a_i)
    elif activation_function == 'ReLU' :
      h_prev = np.maximum(0, a_i)
    h_list.append(h_prev)
  a_i = b_list[num_hidden_layers].flatten()+np.dot(h_prev,W_list[num_hidden_layers].T)
  a_list.append(a_i)
  a_i = a_i-np.max(a_i)
  #calculating softmax  
  a_i = np.where(a_i<-15,-15,a_i)
  a_i = np.where(a_i>15,15,a_i)
  a_i = np.exp(a_i)
  summ = (np.sum((a_i),axis = 1)).reshape(-1,1)
  y = a_i / summ
  return [h_list,a_list,y]



# Initializing weights and biases
def initialized_weights(weight_initializer,num_hidden_layers,hidden_layer_size):
  if weight_initializer == 'random' :
    L = num_hidden_layers+1
    K = hidden_layer_size
    W_list = []
    b_list = []
    W_list.append(np.random.randn(K, 784))
    for i in range(0, num_hidden_layers-1) :
      W_list.append(np.random.randn(K, K))
    W_list.append(np.random.randn(10, K)) 
    for i in range(0,num_hidden_layers) :
      b_list.append(np.random.randn(K, 1))
    b_list.append(np.random.randn(10, 1))
    W_list = np.array(W_list)
    b_list = np.array(b_list)
    return [W_list, b_list]
  elif weight_initializer == 'Xavier' :
    L = num_hidden_layers+1
    K = hidden_layer_size
    W_list = []
    b_list = []
    W_list.append(np.random.uniform(-1.0/28,1.0/28,[K,784]))
    for i in range(0, num_hidden_layers-1) :
      W_list.append(np.random.uniform(-1.0/np.sqrt(K),1.0/np.sqrt(K),[K,K]))
    W_list.append(np.random.uniform(-1.0/np.sqrt(K),1.0/np.sqrt(K),[10,K]))
    for i in range(0,num_hidden_layers) :
      b_list.append(np.zeros((K,1)))
    b_list.append(np.zeros((10,1)))
    W_list = np.array(W_list)
    b_list = np.array(b_list)
    return [W_list, b_list]
  else:
    return initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)

# Compute ouput and loss
def get_loss_value_and_prediction(X,Y,weights,biases,activation_function,num_hidden_layers,hidden_layer_size, loss_type, wt):
  y_pred = feed_forward_neural_network(X,weights,biases,activation_function,num_hidden_layers,hidden_layer_size)[2]
  Loss = 0
  num = X.shape[0]
  forming_Y = []
  for i in range(num):
    e_y = np.zeros((10,))
    e_y[Y[i]] = 1
    forming_Y.append(e_y)
  if (loss_type == 'entropy'):
    for i in range(num):
      for j in range(10):
        if (forming_Y[i][j] == 1):
          if (y_pred[i][j] != 1):
            Loss+=(-1*np.log(y_pred[i][j]))
  else:
    for i in range(num):
      for j in range(10):
        Loss+= (forming_Y[i][j]-y_pred[i][j])**2

  Loss = Loss/num
  #Loss regularization
  for i in range(len(weights)):
    Loss+=(wt/2)*np.sum(np.square(weights[i]))
  for i in range(len(biases)):
    Loss+=(wt/2)*np.sum(np.square(biases[i]))
  
  return [Loss,y_pred]

# Initialize dw,db to 0
def initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size ) :
  K = hidden_layer_size
  #dw
  diff_W_list = []
  diff_W_list.append(np.zeros((K,784)))
  for i in range(0, num_hidden_layers-1) :
    diff_W_list.append(np.zeros((K, K)))
  diff_W_list.append(np.zeros((10, K))) 
  #db
  diff_b_list = []
  for i in range(0,num_hidden_layers) :
    diff_b_list.append(np.zeros((K,1)))
  diff_b_list.append(np.zeros((10,1)))
  W_list = np.array(diff_W_list)
  b_list = np.array(diff_b_list)
  return [W_list, b_list]

# Get g' of a vector
def get_diff_g(h_k, activation_function):
  if activation_function == 'sigmoid' :
      # g'(z) = g(z) * (1 - g(z))
    h_k =  np.multiply(h_k,1-h_k)

  elif activation_function == 'tanh' :
      # g'(z) = 1 - (g(z))^2
    h_k = 1-np.multiply(h_k,h_k)

  elif activation_function == 'ReLU' :
      # g'(z) = 1 if z is positive, 0 otherwise
    h_k =   np.where(h_k>0, 1, 0)
  return (h_k)       


def split_train_valid_data(X_train,Y_train):
    if(len(X_train) == 0):
        A = np.array([])
        return [A,A,A,A]
    size = len(X_train)
    arr = np.arange(size)
    np.random.shuffle(arr)
    train_X = []
    train_Y = []
    # 10% spilt to be considered as validation data
    valid_X = []
    valid_Y = []
    for i in range(size):
      if (10*i<size):
        valid_X.append(X_train[i])
        valid_Y.append(Y_train[i])
      else:
        train_X.append(X_train[i])
        train_Y.append(Y_train[i])
    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    valid_X = np.array(valid_X)
    valid_Y = np.array(valid_Y)
    return [train_X,train_Y,valid_X,valid_Y]

def updated_weights_gd(train_X, train_Y,L, hyper_parameter_combination,initial_weights,initial_biases,loss_type,classes):
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  optimizer = hyper_parameter_combination["optimizer"]
  batch_size = hyper_parameter_combination["batch_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  weight_initializer = hyper_parameter_combination["weight_initializer"]
  activation_function = hyper_parameter_combination["activation_function"]
  print(optimizer)
  if (optimizer == 'vanilla'):
    return do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,len(train_X),L,classes)
  elif (optimizer == 'sgd'):
    return do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,batch_size,L,classes)
  elif (optimizer == 'momentum'):
    return do_momentum(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes)
  elif (optimizer == 'nesterov'):
    return do_nesterov(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes)
  elif (optimizer == 'rmsprop'):
    return do_rmsprop(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes)
  elif (optimizer == 'adam'):
    return do_adam(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes)    
  elif (optimizer == 'nadam'):
    return do_nadam(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes)
  else:
    return do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,batch_size,L,classes)    

def feedforward_with_backpropagation(train_X, train_Y, valid_X,valid_Y, X_test,Y_test, hyper_parameter_combination,loss_type,classes):
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  optimizer = hyper_parameter_combination["optimizer"]
  batch_size = hyper_parameter_combination["batch_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  weight_initializer = hyper_parameter_combination["weight_initializer"]
  activation_function = hyper_parameter_combination["activation_function"]
  #initialize weights
  [initial_weights,initial_biases] = initialized_weights(weight_initializer,num_hidden_layers,hidden_layer_size)
  #use gd and update weights
  L = [valid_X,valid_Y, X_test,Y_test]
  [final_weights,final_biases] = updated_weights_gd(train_X, train_Y,L, hyper_parameter_combination,initial_weights,initial_biases,loss_type,classes)
  #find o/p after updating weights
  return [initial_weights,initial_biases,final_weights,final_biases]

#get accuracy
def get_accuracy(Y_true,Y_pred):
  size = len(Y_true)
  count = 0
  p = int (size)
  for i in range(p):
    max_i = 0
    #index of predicted class
    max_j = 0
    for j in range(10):
      if (Y_pred[i][j]>max_i):
        max_j = j
        max_i = Y_pred[i][j]
    #print(str(max_j)+" "+str(Y_true[i]))
    if (Y_true[i] == max_j):
      count+=1
  return count/size


# to get derivatives of weights and biases
def get_derivatives(W_list,b_list,X,Y,loss_type,num_hidden_layers, hidden_layer_size,activation_function):
  #initializing der_w, der_b to 0
  [der_w,der_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  X.reshape(X.shape[0],784)
  size = X.shape[0]
  #obtaining h, a lists and y_predictions using feed forward neural network
  [h_list,a_list,list_y] = feed_forward_neural_network(X,W_list,b_list,activation_function, num_hidden_layers, hidden_layer_size)
  gradient_a = []
  L = num_hidden_layers+1
  gradient_h = []
  for i in range(L):
    gradient_a.append(0)
  for i in range(L-1):
    gradient_h.append(0)
  gradient_a[L-1] = np.zeros((X.shape[0],10))
  for i in range(X.shape[0]):
    y_pred = list_y[i]
    y_pred.reshape(10,)
    e_y = np.zeros((10,))
    e_y[Y[i]] = 1
    if (loss_type == 'entropy'):
      gradient_a[L-1][i] =  - ( e_y - y_pred ) 
    else:
      diff_a_list = []
      for j in range(10):
        p = 0
        for i in range(10):
          if (i == j):
            p += 2*(y_pred[j]-e_y[j])*(y_pred[j]-y_pred[j]*y_pred[j])
          else:
            p += 2*(y_pred[i]-e_y[i])*(-y_pred[i]*y_pred[j])
        diff_a_list.append(p)
      diff_a_list = np.array(diff_a_list)
      gradient_a[L-1][i] =  diff_a_list
  
  #We got grad_aL(Loss)
  for k in range(L-1,0,-1):
    der_w[k] = np.dot(gradient_a[k].T,h_list[k-1])
    der_b[k] = np.sum(gradient_a[k],axis = 0).reshape(-1,1)
    gradient_h[k-1] = np.dot(gradient_a[k],W_list[k])
    gradient_a[k-1] = np.multiply(gradient_h[k-1],get_diff_g(h_list[k-1], activation_function))

  der_w[0] = np.dot(gradient_a[0].T,X)
  der_b[0] = np.sum(gradient_a[0],axis = 0).reshape(-1,1)
  return [der_w,der_b]

#All gradient descent algos

#Stochastic, Vanilla, Minibatch GD (batch size argument)
def do_sgd(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,batch_size,L,classes):
  [valid_X,valid_Y, X_test,Y_test] = L
  eta = hyper_parameter_combination["eta"]
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  activation_function = hyper_parameter_combination["activation_function"]
  w = initial_weights
  b = initial_biases
  for iter in range(num_epochs):
    size = train_X.shape[0]
    for i in range(0,int(size/batch_size)):
      [dw,db] = get_derivatives(w,b,train_X[i*batch_size:(i+1)*batch_size],train_Y[i*batch_size:(i+1)*batch_size],loss_type,num_hidden_layers, hidden_layer_size,activation_function)
      w_next = w-eta*(dw+weight_decay*w)
      b_next = b-eta*(db+weight_decay*b)       
      w = w_next
      b = b_next
    #logging loses and accuracies  
    loss_params1 = get_loss_value_and_prediction(valid_X,valid_Y,w,b,activation_function,
                                    num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    loss_params2 = get_loss_value_and_prediction(X_test,Y_test,w,b,activation_function,
                                    num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    val_loss = loss_params1[0]
    val_accuracy = get_accuracy(valid_Y,loss_params1[1])
    loss = loss_params2[0]
    accuracy = get_accuracy(Y_test,loss_params2[1])
    to_log = dict()
    to_log["val_loss"] = val_loss
    to_log["val_accuracy"] = val_accuracy
    to_log["accuracy"] = accuracy
    to_log["loss"] = loss
    to_log["epoch"] = iter
    p = np.array(loss_params2[1])
    print(to_log)
    q = draw_confusion_matrix(p, Y_test)
    wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(probs=p,y_true=Y_test,class_names=classes)})
    wandb.log(to_log)
  return [w,b]


#momentum
def do_momentum(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes):
  [valid_X,valid_Y, X_test,Y_test] = L
  eta = hyper_parameter_combination["eta"]
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  activation_function = hyper_parameter_combination["activation_function"]
  batch_size = hyper_parameter_combination["batch_size"]
  w = initial_weights
  b = initial_biases
  gamma = 0.9
  [prev_w, prev_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  for iter in range(num_epochs):
    size = train_X.shape[0]
    for i in range(0,int(size/batch_size)):
      [dw,db] = get_derivatives(w,b,train_X[i*batch_size:(i+1)*batch_size],train_Y[i*batch_size:(i+1)*batch_size],loss_type,num_hidden_layers, hidden_layer_size,activation_function)
      
      min_eta = 0
      w_next = w - (gamma*prev_w+eta*dw+eta*weight_decay*w)
      b_next = b - (gamma*prev_b+eta*db+eta*weight_decay*b)
      prev_w = gamma*prev_w+eta*dw+eta*weight_decay*w
      prev_b = gamma*prev_b+eta*db+eta*weight_decay*b
      w = w_next
      b = b_next
    #logging loses and accuracies    
    loss_params1 = get_loss_value_and_prediction(valid_X,valid_Y,w,b,activation_function,
                                       num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    loss_params2 = get_loss_value_and_prediction(X_test,Y_test,w,b,activation_function,
                                    num_hidden_layers,hidden_layer_size, loss_type,weight_decay) 
    val_loss = loss_params1[0]
    val_accuracy = get_accuracy(valid_Y,loss_params1[1])
    loss = loss_params2[0]
    accuracy = get_accuracy(Y_test,loss_params2[1])
    to_log = dict()
    to_log["val_loss"] = val_loss
    to_log["val_accuracy"] = val_accuracy
    to_log["accuracy"] = accuracy
    to_log["loss"] = loss
    to_log["epoch"] = iter
    print(to_log)
    p = np.array(loss_params2[1])
    q = draw_confusion_matrix(p, Y_test)
    wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(probs=p,y_true=Y_test,class_names=classes)})
    wandb.log(to_log)
  return [w,b]

#nesterov
def do_nesterov(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes):
  [valid_X,valid_Y, X_test,Y_test] = L 
  eta = hyper_parameter_combination["eta"]
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  batch_size = hyper_parameter_combination["batch_size"]
  activation_function = hyper_parameter_combination["activation_function"]
  w = initial_weights
  b = initial_biases
  gamma = 0.9
  [prev_w, prev_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  for iter in range(num_epochs):
    [dw, db] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
    size = train_X.shape[0]
    for i in range(0,int(size/batch_size)):
      v_w = gamma*prev_w
      v_b = gamma*prev_b 
      [dw,db] = get_derivatives(w-v_w,b-v_b,train_X[i*batch_size:(i+1)*batch_size],train_Y[i*batch_size:(i+1)*batch_size],loss_type,num_hidden_layers, hidden_layer_size,activation_function)
      min_eta = 0
      w_next = w - (gamma*prev_w+eta*dw+eta*weight_decay*w)
      b_next = b - (gamma*prev_b+eta*db+eta*weight_decay*b)
      prev_w = gamma*prev_w+eta*dw+eta*weight_decay*w
      prev_b = gamma*prev_b+eta*db+eta*weight_decay*b
      w = w_next
      b = b_next
    #logging loses and accuracies    
    loss_params1 = get_loss_value_and_prediction(valid_X,valid_Y,w,b,activation_function,
                                       num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    loss_params2 = get_loss_value_and_prediction(X_test,Y_test,w,b,activation_function,
                                    num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    val_loss = loss_params1[0]
    val_accuracy = get_accuracy(valid_Y,loss_params1[1])
    loss = loss_params2[0]
    accuracy = get_accuracy(Y_test,loss_params2[1])
    to_log = dict()
    to_log["val_loss"] = val_loss
    to_log["val_accuracy"] = val_accuracy
    to_log["accuracy"] = accuracy
    to_log["loss"] = loss
    to_log["epoch"] = iter
    print(to_log)
    p = np.array(loss_params2[1]) 
    q = draw_confusion_matrix(p, Y_test)
    wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(probs=p,y_true=Y_test,class_names=classes)})
    wandb.log(to_log)
  return [w,b]


#rmsprop
def do_rmsprop(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes):
  [valid_X,valid_Y, X_test,Y_test] = L
  eta = hyper_parameter_combination["eta"]
  batch_size = hyper_parameter_combination["batch_size"]
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  activation_function = hyper_parameter_combination["activation_function"]
  w = initial_weights
  b = initial_biases
  [v_w,v_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  epsilon = 1e-8
  beta = 0.9
  
  for iter in range(num_epochs):
    size = train_X.shape[0]
    for i in range(0,int(size/batch_size)):
      [w_next, b_next] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
      [dw,db] = get_derivatives(w,b,train_X[i*batch_size:(i+1)*batch_size],train_Y[i*batch_size:(i+1)*batch_size],loss_type,num_hidden_layers, hidden_layer_size,activation_function)
      v_w = beta*v_w+(1-beta)*(np.square(dw))
      v_b = beta*v_b+(1-beta)*(np.square(db))
      min_w = w
      min_b = b
      min_eta = 0
      for i in range(w.shape[0]):
        w_next[i] = w[i] - (eta/(np.sqrt(epsilon+v_w[i])))*(dw[i])
      for i in range(b.shape[0]):
        b_next[i] = b[i] - (eta/(np.sqrt(epsilon+v_b[i])))*(db[i])
      w = w_next
      b = b_next
    #logging loses and accuracies    
    loss_params1 = get_loss_value_and_prediction(valid_X,valid_Y,w,b,activation_function,
                                       num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    loss_params2 = get_loss_value_and_prediction(X_test,Y_test,w,b,activation_function,
                                    num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    val_loss = loss_params1[0]
    val_accuracy = get_accuracy(valid_Y,loss_params1[1])
    loss = loss_params2[0]
    accuracy = get_accuracy(Y_test,loss_params2[1])
    to_log = dict()
    to_log["val_loss"] = val_loss
    to_log["val_accuracy"] = val_accuracy
    to_log["accuracy"] = accuracy
    to_log["loss"] = loss
    to_log["epoch"] = iter
    print(to_log)
    p = np.array(loss_params2[1])
    q = draw_confusion_matrix(p, Y_test)
    wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(probs=p,y_true=Y_test,class_names=classes)})
    wandb.log(to_log)
  return [w,b]


#adam
def do_adam(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes):
  [valid_X,valid_Y, X_test,Y_test] = L
  eta = hyper_parameter_combination["eta"]
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  activation_function = hyper_parameter_combination["activation_function"]
  batch_size = hyper_parameter_combination["batch_size"]
  w = initial_weights
  b = initial_biases
  [m_w,m_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  [v_w,v_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  epsilon = 1e-8
  beta1 = 0.9
  beta2 = 0.99
  c = 0
  for iter in range(num_epochs):
    [dw, db] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
    size = train_X.shape[0]
    for i in range(0,int(size/batch_size)):
      c+=1
      [w_next, b_next] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
      [dw,db] = get_derivatives(w,b,train_X[i*batch_size:(i+1)*batch_size],train_Y[i*batch_size:(i+1)*batch_size],loss_type,num_hidden_layers, hidden_layer_size,activation_function)
      m_w = beta1*m_w+(1-beta1)*((dw))
      m_b = beta1*m_b+(1-beta1)*((db))
      v_w = beta2*v_w+(1-beta2)*(np.square(dw))
      v_b = beta2*v_b+(1-beta2)*(np.square(db))
      m_what = m_w/(1-pow(beta1,c+1))
      m_bhat = m_b/(1-pow(beta1,c+1))
      v_what = v_w/(1-pow(beta2,c+1))
      v_bhat = v_b/(1-pow(beta2,c+1))
      min_w = w
      min_b = b
      for i in range(w.shape[0]):
        w_next[i] = w[i] - (eta/(np.sqrt(epsilon+v_what[i])))*(m_what[i])
      for i in range(b.shape[0]):
        b_next[i] = b[i] - (eta/(np.sqrt(epsilon+v_what[i])))*(m_what[i])
      w = w_next
      b = b_next
    #logging loses and accuracies    
    loss_params1 = get_loss_value_and_prediction(valid_X,valid_Y,w,b,activation_function,
                                       num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    loss_params2 = get_loss_value_and_prediction(X_test,Y_test,w,b,activation_function,
                                    num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    val_loss = loss_params1[0]
    val_accuracy = get_accuracy(valid_Y,loss_params1[1])
    loss = loss_params2[0]
    accuracy = get_accuracy(Y_test,loss_params2[1])
    to_log = dict()
    to_log["val_loss"] = val_loss
    to_log["val_accuracy"] = val_accuracy
    to_log["accuracy"] = accuracy
    to_log["loss"] = loss
    to_log["epoch"] = iter
    print(to_log)
    p = np.array(loss_params2[1])
    q = draw_confusion_matrix(p, Y_test)
    wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(probs=p,y_true=Y_test,class_names=classes)})
    wandb.log(to_log)
  return [w,b]

#nadam
def do_nadam(train_X, train_Y,initial_weights,initial_biases,loss_type,hyper_parameter_combination,L,classes):
  [valid_X,valid_Y, X_test,Y_test] = L
  eta = hyper_parameter_combination["eta"]
  num_epochs = hyper_parameter_combination["num_epochs"]
  num_hidden_layers = hyper_parameter_combination["num_hidden_layers"]
  batch_size = hyper_parameter_combination["batch_size"]
  hidden_layer_size = hyper_parameter_combination["hidden_layer_size"]
  weight_decay = hyper_parameter_combination["weight_decay"]
  activation_function = hyper_parameter_combination["activation_function"]
  w = initial_weights
  b = initial_biases
  [m_w,m_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  [v_w,v_b] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
  epsilon = 1e-8
  beta1 = 0.9
  beta2 = 0.99
  c = 0
  for iter in range(num_epochs):
    size = train_X.shape[0]
    for i in range(0,int(size/batch_size)):
      [w_next, b_next] = initialize_diff_weights_and_biases( num_hidden_layers, hidden_layer_size)
      c+=1
      [dw,db] = get_derivatives(w,b,train_X[i*batch_size:(i+1)*batch_size],train_Y[i*batch_size:(i+1)*batch_size],loss_type,num_hidden_layers, hidden_layer_size,activation_function)
      m_w = beta1*m_w+(1-beta1)*((dw))
      m_b = beta1*m_b+(1-beta1)*((db))
      v_w = beta2*v_w+(1-beta2)*(np.square(dw))
      v_b = beta2*v_b+(1-beta2)*(np.square(db))
      m_what = m_w/(1-pow(beta1,c+1))
      m_bhat = m_b/(1-pow(beta1,c+1))
      v_what = v_w/(1-pow(beta2,c+1))
      v_bhat = v_b/(1-pow(beta2,c+1))
      min_w = w
      min_b = b
      for i in range(w.shape[0]):
        w_next[i] = w[i] - (eta/(np.sqrt(epsilon+v_what[i])))*(nest(m_what[i],beta1,c+1,dw[i]))
      for i in range(b.shape[0]):
        b_next[i] = b[i] - (eta/(np.sqrt(epsilon+v_bhat[i])))*(nest(m_bhat[i],beta1,c+1,db[i]))
      w = w_next
      b = b_next
    #logging loses and accuracies    
    loss_params1 = get_loss_value_and_prediction(valid_X,valid_Y,w,b,activation_function,
                                       num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    loss_params2 = get_loss_value_and_prediction(X_test,Y_test,w,b,activation_function,
                                    num_hidden_layers,hidden_layer_size, loss_type,weight_decay)
    val_loss = loss_params1[0]
    val_accuracy = get_accuracy(valid_Y,loss_params1[1])
    loss = loss_params2[0]
    accuracy = get_accuracy(Y_test,loss_params2[1])
    to_log = dict()
    to_log["val_loss"] = val_loss
    to_log["val_accuracy"] = val_accuracy
    to_log["accuracy"] = accuracy
    to_log["loss"] = loss
    to_log["epoch"] = iter
    p = np.array(loss_params2[1])
    q = draw_confusion_matrix(p, Y_test)
    wandb.log({"Confusion Matrix" : wandb.plot.confusion_matrix(probs=p,y_true=Y_test,class_names=classes)})
    wandb.log(to_log)
    print(to_log)
  return [w,b]

def nest(mthat,beta1,t,dw):
  return (beta1*mthat)+(((1-beta1)/(1-pow(beta1,t)))*dw)


# Question 4
sweep_config = {
    'method' : 'random',
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'epochs': {
            'values' : [20,30,40,50,60]
        },
        'num_hidden_layers': {
            'values' : [3,4,5,6]
        },
        'hidden_layer_size': {
            'values' : [16,32,64,128]
        },
        'weight_decay':{
            'values' : [0,0.00005,0.0005,0.005]
        },
        'batch_size' : {
            'values' : [8,16,32,64]
        },
        'optimizer' : {
            'values' : ['momentum','sgd','adam','nadam','rmsprop','nesterov','vanilla']
        },
        'eta' : {
            'values' : [1e-2,1e-3,1e-4,1e-5]  
        },
        'initializer' : {
            'values' : ['random','Xavier']
        },
        'activation_function' : {
            'values' : ['sigmoid','tanh']
        },
    },
}

sweep_id = wandb.sweep(sweep_config, entity="sanjays", project="Dl assignment")

def sweep_train():
  config_defaults = {
        'epochs': 10,
        'num_hidden_layers': 4,
        'hidden_layer_size': 32,
        'weight_decay':0.0005,
        'batch_size' : '16',
        'optimizer' : 'sgd',
        'initializer' : 'random',
        'eta' : 1e-2,
        'activation_function' : 'sigmoid'
  }
  run = wandb.init(config = config_defaults)
  config = wandb.config
  hyper_parameter_combination = {}
  hyper_parameter_combination["num_epochs"] = config.epochs
  hyper_parameter_combination["num_hidden_layers"] = config.num_hidden_layers
  hyper_parameter_combination["hidden_layer_size"] = config.hidden_layer_size
  hyper_parameter_combination["optimizer"] = config.optimizer
  hyper_parameter_combination["batch_size"] = config.batch_size
  hyper_parameter_combination["weight_decay"] = config.weight_decay
  hyper_parameter_combination["weight_initializer"] = config.initializer
  hyper_parameter_combination["activation_function"] = config.activation_function
  hyper_parameter_combination["eta"] = config.eta
  

  (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
  classes = plot_sample_images(X_train, Y_train)
  
  X_train = X_train / 255.0
  X_test = X_test / 255.0
  X_train = X_train.reshape(X_train.shape[0],784)
  X_test = X_test.reshape(X_test.shape[0],784)
  
  
   
  loss_type = 'entropy'


  [train_X,train_Y,valid_X,valid_Y] = split_train_valid_data(X_train,Y_train)
  feedforward_with_backpropagation(train_X, train_Y,valid_X,valid_Y,X_test, Y_test, hyper_parameter_combination,loss_type,classes)
  return


wandb.agent(sweep_id, sweep_train)