
def when_integral_equals_y(fun, dx, y, max_epoch=10 ** 5):
    """ Integrate fun numerically starting from 0, return x when integral reaches target

    Note:
    If a solution can't be found then the integral could not reach the target value if max_epoch steps
    """
    target = y / dx
    s = 0

    for i in range(max_epoch):
        delta_s = (fun(i * dx) + 4 * fun((i+.5) * dx) + fun((i+1) * dx))/6
        new_s = s + delta_s

        if new_s >= target - dx:
            return dx * (i+(target-s)/delta_s)
        else:
            s = new_s
    else:
        raise ValueError('Could not find solution')
