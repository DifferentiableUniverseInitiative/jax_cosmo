# this module stores custom ode code
import jax
import jax.numpy as np


def odeint(fn, y0, t):
    """
    My dead-simple rk4 ODE solver. with no custom gradients
    """

    def rk4(carry, t):
        y, t_prev = carry
        h = t - t_prev
        k1 = fn(y, t_prev)
        k2 = fn(y + h * k1 / 2, t_prev + h / 2)
        k3 = fn(y + h * k2 / 2, t_prev + h / 2)
        k4 = fn(y + h * k3, t)
        y = y + 1.0 / 6.0 * h * (k1 + 2 * k2 + 2 * k3 + k4)
        return (y, t), y

    (yf, _), y = jax.lax.scan(rk4, (y0, np.array(t[0])), t)
    return y
