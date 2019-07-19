import numpy as np
import os
import struct


class MNIST(object):
    def __init__(self,path=["MNIST/train-images.idx3-ubyte","MNIST/train-labels.idx1-ubyte"]):
        path[0] = os.getcwd() + "/" + path[0]
        path[1] = os.getcwd() + "/" + path[1]
        try:
            self.xs_full = self._read_idx(path[0]).reshape(-1,784)/255
            ys_full = self._read_idx(path[1]).reshape(-1,1)
            print(path[0] + ", " + path[1] + " " + "loaded")
        except:
            print("cannot load " + path[0] + ", " + path[1] + ", program will exit")
            exit(-1)
        self.datasize = np.shape(self.xs_full)[0]
        ys_full = np.concatenate((np.arange(self.datasize).reshape(-1, 1), ys_full), axis=1)
        self.ys_full = np.zeros([self.datasize, 10])
        self.ys_full[ys_full[:, 0], ys_full[:, 1]] = 1
        self.pointer = 0

    def _read_idx(self,filename):
        with open(filename, 'rb') as f:
            zero, data_type, dims = struct.unpack('>HBB', f.read(4))
            shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
            return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

    def next_batch(self, batch_size, shuffle=False):
        if shuffle:
            index = np.random.randint(self.datasize, size=batch_size)
            xs = self.xs_full[index, :]
            ys = self.ys_full[index, :]
            return xs, ys
        else:
            if self.pointer + batch_size < self.datasize:
                pass
            else:
                self.pointer = 0
                if batch_size >= self.datasize:
                    batch_size = self.datasize - 1
            xs = self.xs_full[self.pointer:self.pointer + batch_size, :]
            ys = self.ys_full[self.pointer:self.pointer + batch_size, :]
            self.pointer = self.pointer + batch_size
            return xs, ys