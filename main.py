
import scipy_optimize
from scipy import optimize
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler

def run_optimization(eval, grad, w0, check_gradient=False):
    options = dict()
    options['maxiter'] = 1000
    options['disp'] = False
    method = 'L-BFGS-B'

    if check_gradient:
        err = optimize.check_grad(eval, grad, w0)
        print 'Error in gradient: ' + str(err)
    results = optimize.minimize(eval, w0, method=method, jac=grad, options=options)
    w, b = scipy_optimize.unpack_linear(results.x)
    return w, b

def constraints_bound(x, y, x_bound, lower_bound, upper_bound, C, C2):
    transform = StandardScaler()
    x = transform.fit_transform(x)
    opt_data = scipy_optimize.optimize_data(
        x, y, C, C2
    )
    opt_data.x_bound = x_bound
    opt_data.bounds = np.stack((lower_bound, upper_bound), 1)
    eval = scipy_optimize.logistic_bound.create_eval(opt_data)
    grad = scipy_optimize.logistic_bound.create_grad(opt_data)
    w0 = np.zeros(x.shape[1] + 1)
    w0[:] = 0
    return run_optimization(eval, grad, w0)

def constraints_pairwise(x, y, x_low, x_high, C, C2, relative=True, max_delta=None):
    transform = StandardScaler()
    x = transform.fit_transform(x)
    x_low = transform.transform(x_low)
    x_high = transform.transform(x_high)
    opt_data = scipy_optimize.optimize_data(
        x, y, C, C2
    )
    opt_data.x_low = x_low
    opt_data.x_high = x_high
    w0 = np.zeros(x.shape[1] + 1)
    if relative:
        eval = scipy_optimize.logistic_pairwise.create_eval(opt_data)
        grad = scipy_optimize.logistic_pairwise.create_grad(opt_data)
    else:
        opt_data.s = max_delta
        eval = scipy_optimize.logistic_similar.create_eval(opt_data)
        grad = scipy_optimize.logistic_similar.create_grad(opt_data)
    return run_optimization(eval, grad, w0)

def random_xy(w):
    p = w.size
    x = np.random.normal(0, 1, (1, p))
    y = x.dot(w) + np.random.normal(0, 1)
    return x, y

def create_random_bound(w, n, delta):
    p = w.size
    x = np.zeros((n, p))
    upper = np.zeros(n)
    lower = np.zeros(n)
    for i in range(n):
        xi, yi = random_xy(w)
        x[i, :] = xi
        lower[i] = yi - delta
        upper[i] = yi + delta
    return x, lower, upper

def create_random_similar(w, n, max_delta):
    p = w.size
    max_iterations = 10000
    x_similar1 = np.zeros((n, p))
    x_similar2 = np.zeros((n, p))
    iter = 0
    idx = 0
    while iter < max_iterations and idx < n:
        x1, y1 = random_xy(w)
        x2, y2 = random_xy(w)
        iter += 1
        if np.abs(y1-y2) > max_delta:
            continue
        x_similar1[idx, :] = x1
        x_similar2[idx, :] = x2
        idx += 1
    if idx != n:
        print 'Failed to produce enough similar constraints'
    return x_similar1, x_similar2


def create_random_relative(w, n):
    p = w.size
    x_low = np.zeros((n, p))
    x_high = np.zeros((n, p))
    for i in range(n):
        x1, y1 = random_xy(w)
        x2, y2 = random_xy(w)
        if y1 > y2:
            x_high[i, :] = x1
            x_low[i, :] = x2
        else:
            x_high[i, :] = x2
            x_low[i, :] = x1
    return x_high, x_low


if __name__ == '__main__':
    p = 50
    n = 20
    n_mixed = 40
    C = 1
    C2 = 1
    w = np.random.normal(0, 1, p)
    x = np.random.normal(0, 1, (n, p))
    y = x.dot(w) + np.random.normal(0, 1, n)

    x_high, x_low = create_random_relative(w, n_mixed)
    w_relative = constraints_pairwise(x, y, x_low, x_high, C, C2)

    max_delta = .2*np.abs(y.max() - y.min())
    x_similar1, x_similar2 = create_random_similar(w, n_mixed, max_delta)
    w_similar = constraints_pairwise(x, y, x_similar1, x_similar2, C, C2, relative=True, max_delta=max_delta)

    x_bound, lower_bound, upper_bound = create_random_bound(w, n_mixed, y.std())
    w_bound = constraints_bound(x, y, x_bound, lower_bound, upper_bound, C, C2)

    print ''
