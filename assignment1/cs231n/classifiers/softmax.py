import numpy as np
# from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  response = X.dot(W)
  response = response - np.max(response, axis=1).reshape(response.shape[0], 1)
  prob = np.exp(response)
  normalized_prob = prob / np.sum(prob, axis=1).reshape(prob.shape[0], 1)

  evidence = normalized_prob[range(normalized_prob.shape[0]), y]
  loss = -1 * np.sum(np.log(evidence))
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)

  # computing gradient...
  num_train = response.shape[0]
  num_class = W.shape[1]
  prob_sum = np.sum(prob, axis=1)
  for i in range(0, num_train):
      x = X[i: i+1, :].T
      ddW = np.tile(x, (1, num_class))
      ddW = ddW * prob[i, :]
      ddW = ddW / prob_sum[i]

      ddW[:, y[i]: y[i] + 1] -= x

      dW += ddW

  dW /= num_train
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  response = X.dot(W)
  response = response - np.max(response, axis=1).reshape(response.shape[0], 1)
  prob = np.exp(response)
  normalized_prob = prob / np.sum(prob, axis=1).reshape(prob.shape[0], 1)

  evidence = normalized_prob[range(normalized_prob.shape[0]), y]
  loss = -1 * np.sum(np.log(evidence))
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)

  # attempting to vectorize gradient...

  n_prob = normalized_prob
  n_prob[range(0, n_prob.shape[0]), y] -= 1
  dW = X.T.dot(n_prob)
  dW /= X.shape[0]
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

