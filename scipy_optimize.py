import numpy as np
from numpy.linalg import norm
import scipy.optimize as optimize
from scipy.special import expit as sigmoid

def relative_error(v1, v2):
    return norm(v1-v2)/norm(v1)


#from http://stackoverflow.com/questions/4474395/staticmethod-and-abc-abstractmethod-will-it-blend
class abstractstatic(staticmethod):
    __slots__ = ()
    def __init__(self, function):
        super(abstractstatic, self).__init__(function)
        function.__isabstractmethod__ = True
    __isabstractmethod__ = True

def pack_linear(w,b):
    return np.append(w, b)

def unpack_linear(v):
    return v[0:v.size-1], v[v.size-1]

def eval_reg_l2(w):
    return norm(w)**2

def grad_reg_l2(w):
    return 2*w

def loss_l2(y1, y2):
    v = norm(y1-y2)**2
    v_n = v / y1.size
    return v_n
    #return v

def grad_linear_loss_l2(x, y, v):
    w,b = unpack_linear(v)
    n = y.size
    xw = x.dot(w)
    grad_w = 2*(x.T.dot(xw) - x.T.dot(y) + 2*x.T.sum(1)*b)
    grad_b = 2*(xw.sum() - y.sum() + n*b)
    grad_w_n = grad_w/n
    grad_b_n = grad_b/n
    return pack_linear(grad_w_n, grad_b_n)

def apply_linear(x, w, b=None):
    if b is None:
        w,b = unpack_linear(w)
    return x.dot(w) + b

def eval_linear_loss_l2(x, y, v):
    w, b = unpack_linear(v)
    return loss_l2(apply_linear(x, w, b), y)


class optimize_data(object):
    def __init__(self, x, y, reg, reg_mixed):
        self.x = x
        self.y = y
        self.reg = reg
        self.reg_mixed = reg_mixed

    def get_xy(self):
        return self.x, self.y

    def get_reg(self):
        return self.reg, self.reg_mixed


class logistic_optimize(object):
    @abstractstatic
    def eval_mixed_guidance(data, v):
        pass

    @abstractstatic
    def grad_mixed_guidance(data, v):
        pass

    @abstractstatic
    def _grad_num_mixed_guidance(data, v):
        pass

    @classmethod
    def eval(cls, data, v):
        eval_loss = cls.eval_loss(data, v)
        eval_reg = cls.eval_reg(data, v)
        reg, reg_mixed = data.get_reg()
        val = eval_loss + reg*eval_reg
        if reg_mixed > 0:
            eval_mixed = cls.eval_mixed_guidance(data, v)
            val += eval_mixed*reg_mixed
        return val

    @staticmethod
    def eval_loss(data, v):
        x, y = data.get_xy()
        return eval_linear_loss_l2(x, y, v)

    @staticmethod
    def grad_loss(data, v):
        x, y = data.get_xy()
        return grad_linear_loss_l2(x, y, v)

    @staticmethod
    def eval_reg(data, v):
        w, b = unpack_linear(v)
        return eval_reg_l2(w)

    @staticmethod
    def grad_reg(data, v):
        w, b = unpack_linear(v)
        g = grad_reg_l2(w)
        return np.append(g, 0)

    @classmethod
    def grad(cls, data, v):
        grad_loss = cls.grad_loss(data, v)
        reg, reg_mixed = data.get_reg()
        grad_reg = cls.grad_reg(data, v)
        grad_reg *= reg

        I = np.isinf(reg_mixed) | np.isnan(reg_mixed)
        if I.any():
            print 'inf or nan!'
            reg_mixed[I] = 0

        val = grad_loss + grad_reg
        if reg_mixed != 0:
            grad_mixed = cls.grad_mixed_guidance(data, v)
            val += reg_mixed * grad_mixed

        return val

    @classmethod
    def create_eval(cls, data):
        return lambda v: cls.eval(data, v)

    @classmethod
    def create_grad(cls, data):
        return lambda v: cls.grad(data, v)

class logistic_similar(logistic_optimize):
    @staticmethod
    def eval_mixed_guidance(data, v):
        x1 = data.x1
        x2 = data.x2
        s = data.s
        y1 = apply_linear(x1, v)
        y2 = apply_linear(x2, v)
        d = (y2 - y1)
        denom = np.log(1 + np.exp(s+d) + np.exp(d-s) + np.exp(2*d))

        vals = d - denom + np.log(np.exp(s) - np.exp(-s))
        return -vals.sum()


    @staticmethod
    def grad_mixed_guidance(data, v):
        x1 = data.x1
        x2 = data.x2
        s = data.s
        y1 = apply_linear(x1, v)
        y2 = apply_linear(x2, v)
        d = (y2 - y1)
        a = np.exp(s+d) + np.exp(d-s) + np.exp(2*d)
        a2 = np.exp(s+d) + np.exp(d-s) + 2*np.exp(2*d)

        dx = (x2 - x1)
        I = np.isinf(a)
        a[I] = 1
        t = 1 - (a2/(1+a))
        g_fast = (dx.T*t).sum(1)
        g_fast = np.append(g_fast, 0)
        g = g_fast
        return -g

class logistic_pairwise(logistic_optimize):
    @staticmethod
    def eval_mixed_guidance(data, v):
        x_low = data.x_low
        x_high = data.x_high
        if x_low is None:
            return 0
        yj = apply_linear(x_low, v)
        yi = apply_linear(x_high, v)
        d = (yi - yj)
        vals = np.log(1 + np.exp(-d))
        vals_mean = vals.mean()
        return vals_mean


    @staticmethod
    def grad_mixed_guidance(data, v):
        x_low = data.x_low
        x_high = data.x_high
        if x_low is None:
            return 0
        n = x_low.shape[0]
        d = (apply_linear(x_high, v) - apply_linear(x_low, v))
        sig = sigmoid(d)
        g = np.zeros(v.size)

        dx = x_high - x_low
        g_fast = (dx.T*(1-sig)).sum(1)
        g_fast = np.append(g_fast, (1-sig).sum())
        g = g_fast
        g *= -1
        g[-1] *= 0
        g_m = g / n
        return g_m


eps = 1e-2

class logistic_neighbor(logistic_optimize):

    @staticmethod
    def eval_mixed_guidance(data, v):
        assert False, "TODO: Normalize by amount of guidance"
        x = data.x_neighbor
        x_low = data.x_low
        x_high = data.x_high
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low, w, b)
        y_high = apply_linear(x_high, w, b)
        '''
        if (y_low + eps >= y_high).any() or (y + eps >= y_high).any():
            return np.inf
        '''
        sig1 = sigmoid((y_high-y_low))
        sig2 = sigmoid((2*y - y_high - y_low))
        diff = sig1 - sig2
        #assert (np.sign(diff) > 0).all()
        small_constant = getattr(data,'eps',eps)
        #assert False, 'Should this be infinity instead?'
        #diff[diff < 0] = 0
        vals2 = -np.log(diff + small_constant)
        I = np.isnan(vals2)
        if I.any():
            #print 'eval_linear_neighbor_logistic: inf = ' + str(I.mean())
            return np.inf
        val2 = vals2.sum()
        #assert norm(val - val2)/norm(val) < 1e-6
        return val2

    @staticmethod
    def grad_mixed_guidance(data, v):
        assert False, "TODO: Normalize by amount of guidance"
        x = data.x_neighbor
        x_low = data.x_low
        x_high = data.x_high
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low, w, b)
        y_high = apply_linear(x_high, w, b)

        sig1 = sigmoid((y_high-y_low))
        sig2 = sigmoid((2*y - y_high - y_low))
        small_constant = getattr(data,'eps',eps)
        diff = sig1 - sig2
        #diff[diff < 0] = 0
        denom = diff + small_constant
        x1 = (x_high - x_low)
        x2 = (2*x - x_low - x_high)

        num1 = sig1*(1-sig1)
        num2 = sig2*(1-sig2)
        d = x1.T*num1 - x2.T*num2
        g_fast = (d/denom).sum(1)
        g_fast = np.append(g_fast, 0)

        #err = array_functions.relative_error(val, g_fast)
        val = g_fast
        val *= -1
        I = np.isnan(val) | np.isinf(val)
        if I.any():
            #print 'grad_linear_neighbor_logistic: nan!'
            val[I] = 0
        return val

    @staticmethod
    def constraint_neighbor(v, x_low, x_high):
        w,b = unpack_linear(v)
        y_low = apply_linear(x_low,w,b)
        y_high = apply_linear(x_high,w,b)
        return y_high - y_low - eps

    @staticmethod
    def constraint_neighbor2(v, x, x_low, x_high):
        w,b = unpack_linear(v)
        y = apply_linear(x, w, b)
        y_low = apply_linear(x_low,w,b)
        y_high = apply_linear(x_high,w,b)
        return y_high - y - eps

    @staticmethod
    def create_constraint_neighbor(x_low, x_high):
        return lambda v: logistic_neighbor.constraint_neighbor(v, x_low, x_high)

    @staticmethod
    def create_constraint_neighbor2(x, x_low, x_high):
        return lambda v: logistic_neighbor.constraint_neighbor2(v, x, x_low, x_high)


class logistic_bound(logistic_optimize):
    @staticmethod
    def eval_mixed_guidance(data, v):
        w, b = unpack_linear(v)
        x = data.x_bound
        bounds = data.bounds
        y = apply_linear(x, w, b)
        assert y.size == bounds.shape[0]
        c1 = bounds[:, 0]
        c2 = bounds[:, 1]

        sig1 = sigmoid((c2-y))
        sig2 = sigmoid((c1-y))
        small_constant = getattr(data,'eps',eps)
        vals2 = -np.log(sig1-sig2 + small_constant)
        val2 = vals2.mean()
        return val2

    @staticmethod
    def grad_mixed_guidance(data, v):
        bounds = data.bounds
        x = data.x_bound
        w, b = unpack_linear(v)
        y = apply_linear(x, w, b)
        assert y.size == bounds.shape[0]
        c1 = bounds[:, 0]
        c2 = bounds[:, 1]

        sig1 = sigmoid((c2-y))
        sig2 = sigmoid((c1-y))
        small_constant = getattr(data,'eps',eps)
        denom = sig1 - sig2 + small_constant


        num = sig1*(1-sig1) - sig2*(1-sig2)
        num /= denom
        g_fast = (x.T*num).sum(1)
        g_fast = np.append(g_fast, num.sum())
        val = g_fast
        if np.isnan(val).any():
            print 'grad_linear_bound_logistic: nan!'
            val[np.isnan(val)] = 0
        if np.isinf(val).any():
            val[np.isinf(val)] = 0
        val /= x.shape[0]
        return val


