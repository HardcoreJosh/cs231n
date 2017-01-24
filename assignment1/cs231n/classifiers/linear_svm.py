import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero


  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
#################################  my code computing dW
        dW[:, j] += X[i, :].T
        dW[:, y[i]] -= X[i, :].T

  dW /= num_train
  dW += reg * W
#################################

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  #####  half vectorized:
  # loss += 0.5 * reg * np.sum(W * W)
  # delta = 1
  # for i in xrange(X.shape[0]):
  # 	sample = X[i, :]
  # 	diff = delta + sample.dot(W - W[:, y[i]: y[i] + 1])
  # 	loss_i = np.maximum(0, diff)
  # 	loss_i[y[i]] = 0
  # 	loss += np.sum(loss_i)

  # loss /= X.shape[0]

  ####   fully vectorized:
  score_all = X.dot(W)
  score_correct = score_all[range(X.shape[0]), y]
  score_correct = np.reshape(score_correct, (X.shape[0], 1))

  diff = score_all - score_correct
  diff = np.maximum(0, 1 + diff)
  diff[range(X.shape[0]), y] = 0

  loss += np.sum(diff)
  loss /= X.shape[0]
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  for i in xrange(diff.shape[0]):
  	for j in xrange(diff.shape[1]):
  		if diff[i][j] > 0:
  			dW[:, j] += X[i, :].T
  			dW[:, y[i]] -= X[i, :].T

  dW /= X.shape[0]
  dW += reg * W

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
