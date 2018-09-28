def check_shape(tensor, shape, name="Tensor"):
    actual_shape = tuple(tensor.shape)
    actual_dim = len(actual_shape)
    dim = len(shape)
    assert dim == actual_dim, \
        "Expected {name} of dimension {dim}, but got {name} of dim " \
        "{actual_dim}".format(name=name, dim=dim, actual_dim=actual_dim)
    subshape = tuple(axis for axis in shape if axis is not None)
    actual_subshape = tuple(actual_axis for axis, actual_axis in
                            zip(shape, actual_shape) if axis is not None)
    assert actual_subshape == subshape, \
        "Expected {name} of shape {shape}, but got {name} of shape {actual}" \
        "".format(name=name, shape=shape, actual=actual_shape)
