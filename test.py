import numpy as np
import os
import matplotlib.pyplot as plt
import SNN
import data

SAVE_PATH = os.getcwd() + '/weight_mnist'
mnist = data.MNIST(path=["MNIST/t10k-images.idx3-ubyte", "MNIST/t10k-labels.idx1-ubyte"])

w1 = np.load(SAVE_PATH + '1.npy')
w2 = np.load(SAVE_PATH + '2.npy')

Ts = 1e-3
scale = 2
view_max = 2

l1 = SNN.SNNDiscrete(w1, Ts, scale)
l2 = SNN.SNNDiscrete(w2, Ts, scale)

correct = 0

for i in range(mnist.datasize):
    xs, ys = mnist.next_batch(1, shuffle=True)
    xs = (1-xs[0, :])/Ts

    input_mat = np.zeros([784, int(1/Ts*view_max)])
    input_mat[range(784), xs.astype(int)] = 1

    l1out = l1.forward(input_mat)
    l2out = l2.forward(l1out)

    peak = np.argmax(l2out, axis=1)
    prediction = np.argmin(peak)

    label = np.argmax(ys[0])

    if prediction == label:
        correct += 1

    print("test %d" % (i+1))

accuracy = correct / mnist.datasize
print("accuracy = %.4f" % accuracy)
