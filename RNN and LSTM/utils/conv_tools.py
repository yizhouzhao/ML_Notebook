from tabulate import tabulate
from math import ceil

"""
This class provides methods to calculate the input output dimension of a convolutional layer
"""

class ConvCalc:

    @staticmethod
    def conv(input_shape, conv_shape, stride, m=""):
        # input_shape = (H X W X C)
        # conv_shape = (H' X W' X F)
        # Stride = (Hs, Ws)
#         print(input_shape)
        msg = [m]
        H, W, C = input_shape
        H_, W_, F = conv_shape
        Hs, Ws = stride

        H__, W__ =  1 + (H - H_)/Hs, 1 + (W - W_)/Ws
        if (int(H__) != H__) or (int(W__) != W__):
            msg.append("Not Clean Convolution")
            msg.append("H': " + str(H__))
            msg.append("W': " + str(W__))
        output_shape =  (int(H__), int(W__), F)
        return ConvCalc.print_order("Convolution", input_shape, conv_shape+stride,
                   output_shape, msg), output_shape
#         return output_shape

    @staticmethod
    def naive_conv_runtime():
        # Convolutional Runtime by Matrix multiplication
        # TODO naive matrix multiplication
        return

    @staticmethod
    def pool(input_shape, s_shape, stride):
        # input_shape = (F, H X W X C)
        # s_shape = (H' X W')
        # Stride = (Hs, Ws)
        msg = []
        H, W, F = input_shape
        H_, W_ = s_shape
        Hs, Ws = stride

        H__, W__ =  1 + (H - H_)/Hs, 1 + (W - W_)/Ws
        if (int(H__) != H__) or (int(W__) != W__):
            msg.append("Not Clean Pooling")
            msg.append("H': " + str(H__))
            msg.append("W': " + str(W__))
        output_shape =  (ceil(H__), ceil(W__), F)
        return ConvCalc.print_order("Pooling", input_shape, s_shape+stride,
                   output_shape, msg), output_shape
#         return output_shape

    @staticmethod
    def print_order(method, input_shape, f_shape, output_shape, message):
        header = ["input shape", "trans_shape", "output shape", "message"]

        inp = " X ".join([str(i) for i in input_shape])
        fea = " X ".join([str(i) for i in f_shape])
        out = " X ".join([str(i) for i in output_shape])
        msg = " , ".join(message)
        body = []
        body.append(inp)
        body.append(fea)
        body.append(out)
        body.append(msg)
        return header, body
