import numpy as np
import os
import time
import torch


def sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array.
    y is a 1-dimensional int numpy array.
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * y[j]
    return result


def vectorize_sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array. Shape of x is (N, ).
    y is a 1-dimensional int numpy array. Shape of y is (N, ).
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> vectorize_sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    # Write the vecotrized version here
    return np.sum(np.outer(x, y))


def Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
    return result


def vectorize_Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    return np.maximum(x, 0)


def ReluPrime(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result


def vectorize_PrimeRelu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    return np.where(x < 0, 0, 1)


def slice_fixed_point(x, l, start_point):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should have.
    start_point is an integer representing the point at which the final utterance should start in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """

    dim1 = x.shape[0]
    dim2 = l-start_point
    dim3 = x[0].shape[1]

    result = np.array([record[start_point:start_point+l] for record in x])

    return result


def slice_last_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    dim1 = x.shape[0]
    dim2 = l
    dim3 = x[0].shape[1]

    result = np.zeros((dim1, dim2, dim3))

    for i in range(dim1):
        result[i] = x[i][-dim2:]

    return result


def slice_random_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """

    dim1 = x.shape[0]
    dim2 = l
    dim3 = x[0].shape[1]

    offset = [np.random.randint(utter.shape[0]-dim2+1)
              if utter.shape[0]-dim2 > 0 else 0
              for utter in x]

    result = np.array([x[i][offset[i]:offset[i]+dim2] for i in range(dim1)])

    return result


def pad_pattern_end(x):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    dim1 = x.shape[0]
    dim2 = max([utter.shape[0] for utter in x])
    dim3 = x[0].shape[1]

    result = np.zeros((dim1, dim2, dim3))

    modes = ['edge', 'symmetric']
    for i in range(dim1):
        d = len(x[i])
        result[i] = np.pad(x[i], pad_width=(
            (0, dim2-d), (0, 0)), mode=modes[i % len(modes)])

    return result


def pad_constant_central(x, c_):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    dim1 = x.shape[0]
    dim2 = max([utter.shape[0] for utter in x])
    dim3 = x[0].shape[1]

    result = np.zeros((dim1, dim2, dim3))

    for i in range(dim1):
        d = len(x[i])
        result[i] = np.pad(x[i], pad_width=(
            (int((dim2-d)/2), dim2-d-int((dim2-d)/2)), (0, 0)), mode='constant')

    return result


def numpy2tensor(x):
    """
    x is an numpy nd-array.

    Return a pytorch Tensor of the same shape containing the same data.
    """
    return torch.from_numpy(x)


def tensor2numpy(x):
    """
    x is a pytorch Tensor.

    Return a numpy nd-array of the same shape containing the same data.
    """
    return x.numpy()


def tensor_sumproducts(x, y):
    """
    x is an n-dimensional pytorch Tensor.
    y is an n-dimensional pytorch Tensor.

    Return the sum of the element-wise product of the two tensors.
    """
    return torch.sum(torch.mul(x, y))


def tensor_ReLU(x):
    """
    x is a pytorch Tensor.
    For every element i in x, apply the ReLU function:
    RELU(i) = 0 if i < 0 else i

    Return a pytorch Tensor of the same shape as x containing RELU(x)
    """
    return torch.max(x, torch.zeros_like(x))


def tensor_ReLU_prime(x):
    """
    x is a pytorch Tensor.
    For every element i in x, apply the RELU_PRIME function:
    RELU_PRIME(i) = 0 if i < 0 else 1

    Return a pytorch Tensor of the same shape as x containing RELU_PRIME(x)
    """
    return torch.where(x < 0, torch.zeros_like(x), torch.ones_like(x))
