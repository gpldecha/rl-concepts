from tensorflow.python.tools import inspect_checkpoint as chkp

directory = '/home/guillaume/PythonWorkSpace/rl-concepts/rl/models/'
name = 'qmlp.ckpt-6000'

chkp.print_tensors_in_checkpoint_file(directory + name, tensor_name='', all_tensors=True)
