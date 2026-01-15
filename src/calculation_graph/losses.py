import numpy as np

def sum_squared_error(t, y):
    """
    二乗和誤差を計算します。
    $E = \frac{1}{2}\sum(y-t)^2$
    """
    s = ((y - t)**2).sum()
    return s * 0.5