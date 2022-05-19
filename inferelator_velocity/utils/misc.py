def order_dict_to_lists(order_dict):
    """
    Convert dict to two ordered lists
    Used to convert a dict {start_label: (stop_label, start_time, stop_time)}
    into ordered metadata
    """

    # Create a doubly-linked list
    _dll = {}

    for start, (end, _, _) in order_dict.items():

        if start in _dll:

            if _dll[start][1] is not None:
                raise ValueError(f"Both {_dll[start][1]} and {end} follow {start}")

            _dll[start] = (_dll[start][0], end)

        else:

            _dll[start] = (None, end)

        if end in _dll:

            if _dll[end][0] is not None:
                raise ValueError(f"Both {_dll[end][0]} and {start} precede {end}")

            _dll[end] = (start, _dll[end][1])

        else:

            _dll[end] = (start, None)

    _start = None

    for k in order_dict.keys():

        if _dll[k][0] is None and _start is not None:
            raise ValueError("Both {k} and {_start} lack predecessors")

        elif _dll[k][0] is None:
            _start = k

    if _start is None:
        _start = list(order_dict.keys())[0]

    _order = [_start]
    _time = [order_dict[_start][1], order_dict[_start][2]]
    _next = _dll[_start][1]

    while _next is not None and _next != _start:
        _order.append(_next)
        _next = _dll[_next][1]
        if _next in order_dict and _next != _start:
            _time.append(order_dict[_next][1])

    return _order, _time


def vprint(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)
