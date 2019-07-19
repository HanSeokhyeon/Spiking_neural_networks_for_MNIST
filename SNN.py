import numpy as np
import tensorflow as tf
from scipy.signal import lfilter


class SNNLayer(object):
    def __init__(self, in_size, out_size, w=None):

        self.MAX_SPIKE_TIME = 1e5
        self.out_size = out_size
        self.in_size = in_size + 1

        if w is None:
            self.weight = tf.Variable(tf.concat((tf.random_uniform([self.in_size - 1, self.out_size], 0. / self.in_size,
                                                                   8. / self.in_size, tf.float32),
                                                 tf.zeros([1, self.out_size])), axis=0))
        else:
            self.weight = tf.Variable(w, dtype=tf.float32)

    def feedforward(self, layer_in):
        batch_num = tf.shape(layer_in)[0]

        bias_layer_in = tf.ones([batch_num, 1])

        layer_in = tf.concat([layer_in, bias_layer_in], axis=1)

        _, input_sorted_indices = tf.nn.top_k(-layer_in, self.in_size, False)
        input_sorted = tf.batch_gather(layer_in, input_sorted_indices)
        input_sorted_outsize = tf.tile(tf.reshape(input_sorted, [batch_num, self.in_size, 1]), [1, 1, self.out_size])
        weight_sorted = tf.batch_gather(
            tf.tile(tf.reshape(self.weight, [1, self.in_size, self.out_size]), [batch_num, 1, 1]), input_sorted_indices)
        weight_input_mul = tf.multiply(weight_sorted, input_sorted_outsize)
        weight_sumed = tf.cumsum(weight_sorted, axis=1)
        weight_input_sumed = tf.cumsum(weight_input_mul, axis=1)
        out_spike_all = tf.divide(weight_input_sumed, tf.clip_by_value(weight_sumed - 1, 1e-10, 1e10))
        out_spike_ws = tf.where(weight_sumed < 1, self.MAX_SPIKE_TIME * tf.ones_like(out_spike_all), out_spike_all)
        out_spike_large = tf.where(out_spike_ws < input_sorted_outsize,
                                   self.MAX_SPIKE_TIME * tf.ones_like(out_spike_ws), out_spike_ws)
        input_sorted_outsize_slice = tf.slice(input_sorted_outsize, [0, 1, 0],
                                              [batch_num, self.in_size - 1, self.out_size])
        input_sorted_outsize_left = tf.concat(
            [input_sorted_outsize_slice, self.MAX_SPIKE_TIME * tf.ones([batch_num, 1, self.out_size])], 1)
        out_spike_valid = tf.where(out_spike_large > input_sorted_outsize_left,
                                   self.MAX_SPIKE_TIME * tf.ones_like(out_spike_large), out_spike_large)
        out_spike = tf.reduce_min(out_spike_valid, axis=1)
        return out_spike

    def w_sum_cost(self):
        threshold = 1.
        part1 = tf.subtract(threshold, tf.reduce_sum(self.weight, 0))
        part2 = tf.where(part1 > 0, part1, tf.zeros_like(part1))
        return tf.reduce_mean(part2)

    def l2_cost(self):
        w_sqr = tf.square(self.weight)
        return tf.reduce_mean(w_sqr)


class SNNDiscrete(object):
    def __init__(self, w, Ts, scale):
        self.w = w
        self.Ts = Ts
        self.scale = scale
        self.current = np.zeros(int(scale/Ts))
        self.potential = np.zeros(int(scale/Ts))

    def forward(self, x):
        bias = np.zeros([1, np.shape(x)[1]])
        bias[0, 0] = 1
        x = np.append(x, bias, axis=0)
        x_added = np.dot(np.transpose(self.w), x)
        current = lfilter([self.scale], [1, -np.exp(-self.Ts * self.scale)], x_added)
        potential = lfilter([self.Ts], [1, -1], current)
        self.current = current
        self.potential = potential
        spikes_index = np.argmax(potential > 1, axis=1)
        spikes_index[spikes_index == 0] = np.shape(potential)[1] - 1
        spikes = np.zeros_like(potential)
        spikes[np.arange(np.shape(potential)[0]), spikes_index] = 1
        return spikes


def loss_func(both):
    """
    function to calculate loss, refer to paper p.7, formula 11
    :param both: a tensor, it put both layer output and expected output together, its' shape
            is [batch_size,out_size*2], where the left part is layer output(real output), right part is
            the label of input(expected output), the tensor both should be looked like this:
            [[2.13,3.56,7.33,3.97,...0,0,1,0,...]
             [3.14,5.56,2.54,15.6,...0,0,0,1,...]...]
                ↑                   ↑
             layer output           label of input
    :return: a tensor, it is a scalar of loss
    """
    output = tf.slice(both, [0], [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    index = tf.slice(both, [tf.cast(tf.shape(both)[0] / 2, tf.int32)],
                     [tf.cast(tf.shape(both)[0] / 2, tf.int32)])
    z1 = tf.exp(tf.subtract(0., tf.reduce_sum(tf.multiply(output, index))))
    z2 = tf.reduce_sum(tf.exp(tf.subtract(0., output)))
    loss = tf.subtract(
        0., tf.log(
            tf.clip_by_value(tf.divide(
                z1, tf.clip_by_value(
                    z2, 1e-10, 1e10)), 1e-10, 1)))
    return loss