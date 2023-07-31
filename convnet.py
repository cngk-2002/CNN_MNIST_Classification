import cupy as np
from collections import OrderedDict
from layers import *


class ConvNet:
    """Mô hình ConvNet

    conv - relu - pool - affine - relu - dropout - affine - softmax

    Parameters
    ----------
    input_size : Kích thước đầu vào (ví dụ: 784 cho MNIST)
    conv_param : Tham số convolutional layer (filter_num, filter_size, pad, stride)
    hidden_size : Số lượng đơn vị ẩn
    output_size : Kích thước đầu ra (ví dụ: 10 cho MNIST)
    weight_init_std : Độ lệch chuẩn khởi tạo trọng số (ví dụ: 0.01)
    """

    def __init__(self, input_dim=(1, 28, 28), conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = filter_num * int(conv_output_size / 2) * int(conv_output_size / 2)

        # Khởi tạo trọng số
        self.params = {}
        self.params['W1'] = np.sqrt(2 / (input_dim[0] * filter_size * filter_size)) * np.random.randn(filter_num,
                                                                                                      input_dim[0],
                                                                                                      filter_size,
                                                                                                      filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = np.sqrt(2 / pool_output_size) * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = np.sqrt(2 / hidden_size) * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        # Tạo các layer
        self.layers = OrderedDict()
        self.layers['Conv'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Dropout'] = Dropout()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        """Tính giá trị hàm mất mát

        Parameters
        ----------
        x : Dữ liệu đầu vào
        t : Nhãn đúng

        Returns
        -------
        Giá trị hàm mất mát
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        """Tính độ chính xác

        Parameters
        ----------
        x : Dữ liệu đầu vào
        t : Nhãn đúng

        Returns
        -------
        Độ chính xác
        """
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """Tính gradient

        Parameters
        ----------
        x : Dữ liệu đầu vào
        t : Nhãn đúng

        Returns
        -------
        Gradient của các trọng số
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Gradients
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv'].dW, self.layers['Conv'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
