from typing import Any, Counter, DefaultDict, Dict, Optional, Tuple, Union
from functools import partial
import typing
from collections import Counter, OrderedDict
import numpy as np
from numpy import prod
from itertools import zip_longest

from fvcore.nn.jit_handles import (
    Handle,
    addmm_flop_jit,
    batchnorm_flop_jit,
    bmm_flop_jit,
    conv_flop_jit,
    einsum_flop_jit,
    elementwise_flop_counter,
    linear_flop_jit,
    matmul_flop_jit,
    norm_flop_counter,
)

def get_shape(val: object) -> typing.List[int]:
    """
    Get the shapes from a jit value object.
    Args:
        val (torch._C.Value): jit value object.
    Returns:
        list(int): return a list of ints.
    """
    if val.isCompleteTensor():  # pyre-ignore
        r = val.type().sizes()  # pyre-ignore
        if not r:
            r = [1]
        return r
    elif val.type().kind() in ("IntType", "FloatType"):
        return [1]
    else:
        raise ValueError()

def basic_binary_op_flop_jit(inputs, outputs, name):
    input_shapes = [get_shape(v) for v in inputs]
    # for broadcasting
    input_shapes = [s[::-1] for s in input_shapes]
    max_shape = np.array(list(zip_longest(*input_shapes, fillvalue=1))).max(1)
    flop = prod(max_shape)
    flop_counter = Counter({name: flop})
    return flop_counter

def rsqrt_flop_jit(inputs, outputs):
    input_shapes = [get_shape(v) for v in inputs]
    flop = prod(input_shapes[0]) * 2
    flop_counter = Counter({"rsqrt": flop})
    return flop_counter

def dropout_flop_jit(inputs, outputs):
    input_shapes = [get_shape(v) for v in inputs[:1]]
    flop = prod(input_shapes[0])
    flop_counter = Counter({"dropout": flop})
    return flop_counter

def softmax_flop_jit(inputs, outputs):
    # from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/profiler/internal/flops_registry.py
    input_shapes = [get_shape(v) for v in inputs[:1]]
    flop = prod(input_shapes[0]) * 5
    flop_counter = Counter({'softmax': flop})
    return flop_counter

def _reduction_op_flop_jit(inputs, outputs, reduce_flops=1, finalize_flops=0):
    input_shapes = [get_shape(v) for v in inputs]
    output_shapes = [get_shape(v) for v in outputs]

    in_elements = prod(input_shapes[0])
    out_elements = prod(output_shapes[0])

    num_flops = (in_elements * reduce_flops
        + out_elements * (finalize_flops - reduce_flops))

    return num_flops

def conv_flop_count(
    x_shape: typing.List[int],
    w_shape: typing.List[int],
    out_shape: typing.List[int],
) -> typing.Counter[str]:
    """
    This method counts the flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
    Returns:
        Counter: A Counter dictionary that records the number of flops for each
            operation.
    """
    batch_size, Cin_dim, Cout_dim = x_shape[0], w_shape[1], out_shape[1]
    out_size = prod(out_shape[2:])
    kernel_size = prod(w_shape[2:])
    flop = batch_size * out_size * Cout_dim * Cin_dim * kernel_size
    flop_counter = Counter({"conv": flop})
    return flop_counter


# A dictionary that maps supported operations to their flop count jit handles.
_DEFAULT_SUPPORTED_OPS: Dict[str, Handle] = {
    "aten::addmm": addmm_flop_jit,
    "aten::bmm": bmm_flop_jit,
    "aten::_convolution": conv_flop_jit,
    "aten::einsum": einsum_flop_jit,
    "aten::matmul": matmul_flop_jit,
    "aten::mm": matmul_flop_jit,
    "aten::linear": linear_flop_jit,
    # You might want to ignore BN flops due to inference-time fusion.
    # Use `set_op_handle("aten::batch_norm", None)
    "aten::batch_norm": batchnorm_flop_jit,
    "aten::group_norm": norm_flop_counter(2),
    "aten::layer_norm": norm_flop_counter(2),
    "aten::instance_norm": norm_flop_counter(1),
    "aten::upsample_nearest2d": elementwise_flop_counter(0, 1),
    "aten::upsample_bilinear2d": elementwise_flop_counter(0, 4),
    "aten::adaptive_avg_pool2d": elementwise_flop_counter(1, 0),
    "aten::grid_sampler": elementwise_flop_counter(0, 4),  # assume bilinear

    "aten::add": partial(basic_binary_op_flop_jit, name='aten::add'),
    "aten::add_": partial(basic_binary_op_flop_jit, name='aten::add_'),
    "aten::mul": partial(basic_binary_op_flop_jit, name='aten::mul'),
    "aten::sub": partial(basic_binary_op_flop_jit, name='aten::sub'),
    "aten::div": partial(basic_binary_op_flop_jit, name='aten::div'),
    "aten::floor_divide": partial(basic_binary_op_flop_jit, name='aten::floor_divide'),
    "aten::relu": partial(basic_binary_op_flop_jit, name='aten::relu'),
    "aten::relu_": partial(basic_binary_op_flop_jit, name='aten::relu_'),
    "aten::gelu": partial(basic_binary_op_flop_jit, name='aten::gelu'),
    "aten::rsqrt": rsqrt_flop_jit,
    "aten::softmax": softmax_flop_jit,
    "aten::dropout": dropout_flop_jit,
}

def val_shapes_first_100():
    return [[3, 800, 1201], [3, 873, 800], [3, 800, 1060], [3, 1066, 800], [3, 1196, 800], [3, 800, 1204], [3, 1207, 800], [3, 824, 800], [3, 800, 1199], 
        [3, 800, 1066], [3, 800, 1199], [3, 1199, 800], [3, 1066, 800], [3, 800, 1000], [3, 656, 1332], [3, 800, 1066], [3, 800, 1066], [3, 800, 800], [3, 800, 1066], 
        [3, 1199, 800], [3, 800, 1204], [3, 800, 1221], [3, 800, 1066], [3, 800, 1199], [3, 800, 1066], [3, 800, 1199], [3, 800, 1199], [3, 800, 1324], [3, 1120, 800], 
        [3, 800, 1199], [3, 1066, 800], [3, 800, 1066], [3, 762, 1332], [3, 800, 922], [3, 800, 1066], [3, 1155, 800], [3, 756, 1332], [3, 800, 800], [3, 800, 1199], 
        [3, 800, 1333], [3, 800, 1066], [3, 1066, 800], [3, 800, 1204], [3, 1204, 800], [3, 800, 1066], [3, 800, 800], [3, 800, 1066], [3, 800, 1066], [3, 800, 1204], 
        [3, 1066, 800], [3, 800, 1204], [3, 727, 1333], [3, 1201, 800], [3, 1153, 800], [3, 800, 1066], [3, 752, 1333], [3, 800, 1196], [3, 800, 964], [3, 731, 1332], 
        [3, 800, 1204], [3, 800, 1199], [3, 800, 1201], [3, 800, 1010], [3, 752, 1333], [3, 1066, 800], [3, 800, 1199], [3, 1199, 800], [3, 800, 1066], [3, 800, 1066], 
        [3, 1071, 800], [3, 800, 1201], [3, 800, 1062], [3, 752, 1333], [3, 800, 1200], [3, 800, 1066], [3, 800, 1066], [3, 800, 1066], [3, 800, 1199], [3, 800, 1199], 
        [3, 800, 1199], [3, 802, 800], [3, 1193, 800], [3, 748, 1333], [3, 800, 1066], [3, 800, 1115], [3, 800, 800], [3, 800, 1201], [3, 800, 800], [3, 800, 1066], 
        [3, 800, 1199], [3, 800, 1201], [3, 800, 949], [3, 800, 1200], [3, 800, 1066], [3, 929, 800], [3, 800, 1066], [3, 800, 1199], [3, 800, 1066], [3, 930, 800], [3, 800, 1066]]