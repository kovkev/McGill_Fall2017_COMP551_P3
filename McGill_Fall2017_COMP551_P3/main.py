import numpy as np
import math

def dprint(*args, **kwargs):
  return
  print(args, kwargs)

class Layer(object):
  def __init__(self, num_input_nodes, num_output_nodes, activation_function):
    self.num_input_nodes = num_input_nodes
    self.num_output_nodes = num_output_nodes
    self.into = [0] * num_input_nodes
    self.activation_function = activation_function

  def initialize_weights(self):
    self.weights = []

    for i in range(self.num_input_nodes):
      row = []

      for j in range(self.num_output_nodes):
        weight_value = self.sample_standard_normal()

        row.append(weight_value)

      self.weights.append(row)

    self.biases = np.array([
      self.sample_standard_normal() for i in range(self.num_output_nodes)
    ]) # adding a bias for each node

    self.weights = np.array(self.weights)

    self.zs = np.array([0] * self.num_output_nodes)

  def sample_standard_normal(self):
    return np.random.normal()

  def compute_output(self):
    # for i in range(self.num_output_nodes):
    #   print(self.into)
    #   print(self.weights)
    into = np.concatenate(
      (
        self.into,
      ),
    )

    self.zs = np.dot(
      self.weights.T,
      into,
    ) + self.biases

    output = self.activation_function(self.zs)

    return output

  def representation(self):
    s = ""
    # for i in range(self.num_input_nodes):
    #   s += "row:"
    #   for j in range(self.num_output_nodes):
    #     s += " " + str(self.weights[i][j]) + " "
    #   s += "\n"
    s += str(self.weights)
    s += "\n"

    s += str(self.biases)
    s += "\n"

    return s


def sigmoid(x):
  return 1/(1 + math.e**(-x))

def sigmoid_prime(x):
  s = sigmoid(x)
  return s * (1 - s)

def linear(x):
  return x

def linear_prime(x):
  return x/x

derivatives = {}
derivatives[sigmoid] = sigmoid_prime
derivatives[linear] = linear_prime

class FFNN(object):
  def __init__(self, layer_sizes=[1], learning_rate=0.01, activation_function=None):
    self.learning_rate = learning_rate
    self.activation_function = activation_function or linear

    self.layers = []
    self.layers.append(Layer(0, layer_sizes[0], self.activation_function))

    for i in range(1, len(layer_sizes)):
      layer_input_size = layer_sizes[i - 1]
      layer_output_size = layer_sizes[i]
      self.layers.append(Layer(layer_input_size, layer_output_size, self.activation_function))

    for layer in self.layers:
      layer.initialize_weights()

  def representation(self):
    s = ""
    for i in range(len(self.layers)):
      layer = self.layers[i]
      s += "layer %s " % i + ":\n"
      s += layer.representation()
      s += "\n\n"
    return s

  def error(self, example):
    dprint(example)
    features, result = example

    features = np.array(features)
    result = np.array(result)

    self.layers[0].compute_output()
    self.layers[0].output = features

    for i in range(1, len(self.layers)):
      # print("layer %s " % i )
      self.layers[i].into = self.layers[i-1].output
      # print(self.layers[i].into )
      self.layers[i].output = self.layers[i].compute_output()
      # print(self.layers[i].output )

    error = self.layers[-1].output - result
    return error

  def train(self, example):
    error = self.error(example)

    # d_a_C = error
    deltas = [0] * (len(self.layers))
    deltas[-1] = np.multiply(
      error,
      derivatives[self.activation_function]( self.layers[-1].zs )
    )

    # dc_da[ (l, j) ] = dC/da_j^l
    # dc_da = {}
    # dc_da[ (len(self.layers), 0) ] = error

    for i in reversed(range(0, len(self.layers) - 1)):
      dprint(deltas[i + 1])
      dprint("i : %s" % i)
      dprint("w -> m : %s" %  str(self.layers[i+1].weights.shape))
      deltas[i] = np.multiply(
        np.dot(
          self.layers[i+1].weights,
          deltas[i+1]
        ),
        np.concatenate(
          (
            derivatives[self.activation_function]( self.layers[i].zs ),
            np.array([]),
          )
        )
      )

      # dc_da[i] = 0
      # for j in range(len(self.layers[i])):
      #   dc_da[i] += self.layers[i][j] * sigma_p[] * dc_da[i+1]

      # dc_dw = a[l-1][k] * sigma_p[i] * dc_dc[i]

      # for j in range(len(self.layers[i])):
      #   layers[i].weights[j] -= dc_dw

    for i in reversed(range(1, len(self.layers))):
      dprint(">>>")
      dprint(i)
      layer = self.layers[i]

      # a_l-1
      dprint(layer.weights.shape)
      a_prev = self.layers[i-1].output.T
      # print(a_prev)
      # print(deltas[i])
      asiz = len(a_prev)
      dsiz = len(deltas[i])
      mul = (
        a_prev *
        deltas[i].reshape(dsiz, 1)
      ).T

      layer.weights = (
        layer.weights
        - self.learning_rate
        * mul
      )
      layer.biases = (
        layer.biases
        - self.learning_rate
        * deltas[i]
      )

def get_dataset():
  dataset = [
    (
      [5],
      [6],
    )
  ]
  validate_data(dataset)
  return dataset
  # with open() as f:
  #   pass

def validate_data(dataset):
  feature_size = len(dataset[0][0])
  result_size = len(dataset[0][1])

  for example in dataset:
    assert len(example[0]) == feature_size
    assert len(example[1]) == result_size

def main():
  dataset = get_dataset()

  layer_sizes=[
    1,
    3,
    3,
    1,
  ]

  learning_rate=0.001

  ffnn = FFNN(
    layer_sizes=layer_sizes,
    learning_rate=learning_rate,
    # activation_function=linear,
  )

  assert len(dataset[0][0]) == layer_sizes[0]
  assert len(dataset[0][1]) == layer_sizes[-1]

  print(ffnn.representation())
  for epoch_t in range(500):
    for example in dataset:
      dprint(example)
      ffnn.train(example)

    # training_error()
    for example in dataset:
      error = ffnn.error(example)
      print(sum(error**2))
    # validation_error()
  print(ffnn.representation())

if __name__ == "__main__":
  main()

# hyperparamets
# outputs / accuracy
# confusion matrix
#   cm[m,n] = percetage of times m is predicted as n
# runtimes
# validation test error vs test set error over epoch (for best model)
#   for each epoch
#   cross-validation

# 5 models
#  each model has 1 partition as its validation set for all epoch
#  each model has different hyperparameters
#  try simple hyperparameter differences
