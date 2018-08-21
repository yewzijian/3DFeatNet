import time
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def time_function(some_function):
    """
    Outputs the time a function takes
    to execute.
    """
    def wrapper():
        t1 = time.time()
        some_function()
        t2 = time.time()
        return "Time it took to run the function: " + str((t2 - t1)) + "\n"
    return wrapper


def get_tensors_in_checkpoint_file(file_name):
    """Get tensors in a checkpoint file.
    If no `tensor_name` is provided, prints the tensor names and shapes
    in the checkpoint file.
    If `tensor_name` is provided, prints the content of the tensor.
    Args:
    file_name: Name of the checkpoint file.
    """
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)

    var_list = list(reader.get_variable_to_shape_map().keys())

    return var_list


if __name__ == '__main__':
    get_tensors_in_checkpoint_file('/home/zyew3/projects/pointcloud_alignment_log/20180209_091008/ckpt/checkpoint.ckpt-56000')