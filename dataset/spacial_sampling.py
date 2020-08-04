import numpy as np

def sample(a, b):
	return (b - a) * np.random.random() + a

def sample_n(a, b, N):
	return (b - a) * np.random.random(N) + a


def bmm(A, B):

    ret = []
    for a, b in zip(A, B):
        ret.append(np.dot(a, b))
    
    return np.array(ret)


def get_rotation_matrix(sx, sy, sz):

    rx = np.zeros((sx.shape[0], 3, 3))
    ry = np.zeros((sy.shape[0], 3, 3))
    rz = np.zeros((sz.shape[0], 3, 3))

    rx[:, 0, 0] = 1.
    rx[:, 1, 1] = np.cos(sx)
    rx[:, 1, 2] = -np.sin(sx)
    rx[:, 2, 1] = np.sin(sx)
    rx[:, 2, 2] = np.cos(sx)

    ry[:, 1, 1] = 1.
    ry[:, 0, 0] = np.cos(sy)
    ry[:, 0, 2] = np.sin(sy)
    ry[:, 2, 0] = -np.sin(sy)
    ry[:, 2, 2] = np.cos(sy)

    rz[:, 0, 0] = np.cos(sz)
    rz[:, 0, 1] = -np.sin(sz)
    rz[:, 1, 0] = np.sin(sz)
    rz[:, 1, 1] = np.cos(sz)
    rz[:, 2, 2] = 1.

    #ryrx = np.einsum('bij,bkl->bil', ry, rx)
    #rot = np.einsum('bij,bkl->bil', rz, ryrx)
    #ryrx = np.einsum('bij,bjk->bjk', ry, rx)
    #rot = np.einsum('bij,bjk->bjk', rz, ryrx)
    #ryrx = np.einsum('bij,bkl->bik', ry, rx)
    #rot = np.einsum('bij,bkl->bik', rz, ryrx)
    ryrx = bmm(ry, rx)
    rot = bmm(rz, ryrx)
    #print(ryrx.shape)
    #print(rot.shape)

    return rot


class base():
    def __init__(self):
        pass

class T_perspective(base):
    def __init__(self, cfg):
        super(T_perspective, self).__init__()

        self.x_min_near = None
        self.x_max_near = None
        self.x_min_far = None
        self.x_max_far = None
        self.y_min_far = None
        self.y_max_far = None
        self.y_min_near = None
        self.y_max_near = None
        self.z_min = None
        self.z_max = None

        #print(cfg)
        if type(cfg['x_near']) is list and type(cfg['x_far']) is list:
            self.x_min_near = cfg['x_near'][0]
            self.x_max_near = cfg['x_near'][1]
            self.x_min_far = cfg['x_far'][0]
            self.x_max_far = cfg['x_far'][1]
            self.get_x = self._get_x_perspective
        else:
            self.get_x = self._get_zero
        if type(cfg['y_near']) is list and type(cfg['y_far']) is list:
            self.y_min_near = cfg['y_near'][0]
            self.y_max_near = cfg['y_near'][1]
            self.y_min_far = cfg['y_far'][0]
            self.y_max_far = cfg['y_far'][1]
            self.get_y = self._get_y_perspective
        else:
            self.get_y = self._get_zero

        if type(cfg['z']) is list:
            self.z_min = cfg['z'][0]
            self.z_max = cfg['z'][1]
            self.get_z = self._get_z_perspective
        else:
            self.get_z = self._get_zero

    def _get_x_perspective(self, z):
        x_min = self.x_min_near * (1. - z) + self.x_min_far * z
        x_max = self.x_max_near * (1. - z) + self.x_max_far * z
        x = np.array([sample(xmin, xmax) for xmin, xmax in zip(x_min, x_max)])
        return x

    def _get_y_perspective(self, z):
        y_min = self.y_min_near * (1. - z) + self.y_min_far * z
        y_max = self.y_max_far * (1. - z) + self.y_max_far * z
        y = np.array([sample(xmin, xmax) for xmin, xmax in zip(y_min, y_max)])
        return y

    def _get_z_perspective(self, z):
        z = (self.z_max - self.z_min) * z + self.z_min
        return z

    def _get_zero(self, z):
        x = 0.0 * np.copy(z)
        return x
        
    def get(self, batchsize):

        _z = sample_n(0., 1., batchsize)
        x = self.get_x(_z)
        y = self.get_y(_z)
        z = self.get_z(_z)
 
        return np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), 1)



class T_const(base):
    def __init__(self, cfg):
        super(T_const, self).__init__()
        self._x = cfg['x']
        self._y = cfg['y']
        self._z = cfg['z']

    def _get_const(self, c):
        return 

    def get(self, batchsize):

        x = self._x * np.ones(batchsize)
        y = self._y * np.ones(batchsize)
        z = self._z * np.ones(batchsize)
 
        return np.concatenate((x[:, np.newaxis], y[:, np.newaxis], z[:, np.newaxis]), 1)



class R_random(base):
    def __init__(self, cfg):
        super(R_random, self).__init__()

        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.z_min = None
        self.z_max = None
        #print(cfg)
        if type(cfg['rx']) is list:
            self.x_min = cfg['rx'][0]
            self.x_max = cfg['rx'][1]
        if type(cfg['ry']) is list:
            self.y_min = cfg['ry'][0]
            self.y_max = cfg['ry'][1]
        if type(cfg['rz']) is list:
            self.z_min = cfg['rz'][0]
            self.z_max = cfg['rz'][1]

    def get_rot(self, batchsize, _max, _min):
        if _min is None and _max is None:
            rr = np.zeros(batchsize)
        else:
            rr = sample_n(_min, _max, batchsize)
        
        #print(rr)
        return rr

    def get(self, batchsize):
        
        rx = self.get_rot(batchsize, self.x_max, self.x_min)
        ry = self.get_rot(batchsize, self.y_max, self.y_min)
        rz = self.get_rot(batchsize, self.z_max, self.z_min)
            
        return get_rotation_matrix(rx, ry, rz)


class R_const(base):
    def __init__(self, cfg):
        super(R_const, self).__init__()
        self._x = np.deg2rad(cfg['rx'])
        self._y = np.deg2rad(cfg['ry'])
        self._z = np.deg2rad(cfg['rz'])

    def get(self, batchsize):
        
        rx = self._x * np.ones(batchsize)
        ry = self._y * np.ones(batchsize)
        rz = self._z * np.ones(batchsize)
            
        return get_rotation_matrix(rx, ry, rz)


