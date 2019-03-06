import cntk
from cntk.layers.blocks import *
from cntk.internal import _as_tuple

_INFERRED = cntk.layers.blocks._INFERRED

def CIFG_LSTM_(shape, cell_shape, activation, use_peepholes,
                    init, init_bias,
                    enable_self_stabilization,
                    name=''):
    has_projection = cell_shape is not None

    shape = _as_tuple(shape)

    cell_shape = _as_tuple(cell_shape) if cell_shape is not None else shape
    if len(shape) != 1 or len(cell_shape) != 1:
        raise ValueError("%s: shape and cell_shape must be vectors (rank-1 tensors)" % type)
    stack_axis = -1      
    cell_shape_list = list(cell_shape)
    stacked_dim = cell_shape_list[stack_axis]
    type = 'LSTM'
    cell_shape_list[stack_axis] = stacked_dim * 4
    cell_shape_stacked = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times
    cell_shape_list[stack_axis] = stacked_dim * 4
    cell_shape_stacked_H = tuple(cell_shape_list)  # patched dims with stack_axis duplicated 4 times

    # parameters
    b  = Parameter(            cell_shape_stacked,   init=init_bias, name='b')                              # bias
    W  = Parameter(_INFERRED + cell_shape_stacked,   init=init,      name='W')                              # input
    H  = Parameter(shape     + cell_shape_stacked_H, init=init,      name='H')                              # hidden-to-hidden
    H1 = Parameter(shape     + cell_shape,           init=init,      name='H1') if type == 'GRU' else None  # hidden-to-hidden
    Ci = Parameter(            cell_shape,           init=init,      name='Ci') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Cf = Parameter(            cell_shape,           init=init,      name='Cf') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}
    Co = Parameter(            cell_shape,           init=init,      name='Co') if use_peepholes else None  # cell-to-hiddden {note: applied elementwise}

    Wmr = Parameter(cell_shape + shape, init=init, name='P') if has_projection else None  # final projection

    # each use of a stabilizer layer must get its own instance
    Sdh = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dh_stabilizer')
    Sdc = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='dc_stabilizer')
    Sct = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='c_stabilizer')
    Sht = Stabilizer(enable_self_stabilization=enable_self_stabilization, name='P_stabilizer')

    # LSTM model function
    # in this case:
    #   (dh, dc, x) --> (h, c)
    def lstm(dh, dc, x):
        dhs = Sdh(dh)  # previous values, stabilized
        dcs = Sdc(dc)
        # note: input does not get a stabilizer here, user is meant to do that outside

        # projected contribution from input(s), hidden, and bias
        proj4 = b + times(x, W) + times(dhs, H)

        it_proj  = slice (proj4, stack_axis, 0*stacked_dim, 1*stacked_dim)  # split along stack_axis
        bit_proj = slice (proj4, stack_axis, 1*stacked_dim, 2*stacked_dim)
        ft_proj  = slice (proj4, stack_axis, 2*stacked_dim, 3*stacked_dim)
        ot_proj  = slice (proj4, stack_axis, 3*stacked_dim, 4*stacked_dim)

        # helper to inject peephole connection if requested
        def peep(x, c, C):
            return x + C * c if use_peepholes else x

        it = sigmoid (peep (it_proj, dcs, Ci))        # input gate(t)
        # TODO: should both activations be replaced?
        bit = it * activation (bit_proj)              # applied to tanh of input network

        ft = 1 - it # MODIFIED
        bft = ft * dc                                 # applied to cell(t-1)

        ct = bft + bit                                # c(t) is sum of both

        ot = sigmoid (peep (ot_proj, Sct(ct), Co))    # output gate(t)
        ht = ot * activation (ct)                     # applied to tanh(cell(t))

        c = ct                                        # cell value
        h = times(Sht(ht), Wmr) if has_projection else \
            ht

        # returns the new state as a tuple with names but order matters
        #return (Function.NamedOutput(h=h), Function.NamedOutput(c=c))
        return (h, c)
    return BlockFunction(type, name)(lstm)


def CIFG_LSTM(shape, cell_shape=None, activation=default_override_or(tanh), use_peepholes=default_override_or(False),
         init=default_override_or(glorot_uniform()), init_bias=default_override_or(0),
         enable_self_stabilization=default_override_or(False),
         name=''):
    '''
    LSTM(shape, cell_shape=None, activation=tanh, use_peepholes=False, init=glorot_uniform(), init_bias=0, enable_self_stabilization=False, name='')
    Layer factory function to create an LSTM block for use inside a recurrence.
    The LSTM block implements one step of the recurrence and is stateless. It accepts the previous state as its first two arguments,
    and outputs its new state as a two-valued tuple ``(h,c)``.
    Example:
     >>> # a typical recurrent LSTM layer
     >>> from cntk.layers import *
     >>> lstm_layer = Recurrence(LSTM(500))
    Args:
        shape (`int` or `tuple` of `ints`): vector or tensor dimension of the output of this layer
        cell_shape (tuple, defaults to `None`): if given, then the output state is first computed at `cell_shape`
         and linearly projected to `shape`
        activation (:class:`~cntk.ops.functions.Function`, defaults to :func:`~cntk.ops.tanh`): function to apply at the end, e.g. `relu`
        use_peepholes (bool, defaults to `False`):
        init (scalar or NumPy array or :mod:`cntk.initializer`, defaults to `glorot_uniform`): initial value of weights `W`
        init_bias (scalar or NumPy array or :mod:`cntk.initializer`, defaults to 0): initial value of weights `b`
        enable_self_stabilization (bool, defaults to `False`): if `True` then add a :func:`~cntk.layers.blocks.Stabilizer`
         to all state-related projections (but not the data input)
        name (str, defaults to ''): the name of the Function instance in the network
    Returns:
        :class:`~cntk.ops.functions.Function`:
        A function ``(prev_h, prev_c, input) -> (h, c)`` that implements one step of a recurrent LSTM layer.
    '''

    activation                = get_default_override(LSTM, activation=activation)
    use_peepholes             = get_default_override(LSTM, use_peepholes=use_peepholes)
    init                      = get_default_override(LSTM, init=init)
    init_bias                 = get_default_override(LSTM, init_bias=init_bias)
    enable_self_stabilization = get_default_override(LSTM, enable_self_stabilization=enable_self_stabilization)

    return CIFG_LSTM_(shape, cell_shape, activation=activation, use_peepholes=use_peepholes,
                           init=init, init_bias=init_bias,
                           enable_self_stabilization=enable_self_stabilization, name=name)