import argparse
#import torch
#import cv2
#import select_dataset
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
from numpy import random

if __name__ == '__main__':
    import spacial_sampling, data_loader
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from utils import utils  
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset import spacial_sampling, data_loader
    from utils import utils  

def _create_raymond_lights(c, intensity):
    
	thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
	phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

	nodes = []
	for phi, theta in zip(phis, thetas):
		xp = np.sin(theta) * np.cos(phi)
		yp = np.sin(theta) * np.sin(phi)
		zp = np.cos(theta)

		z = np.array([xp, yp, zp])
		z = z / np.linalg.norm(z)
		x = np.array([-z[1], z[0], 0.0])
		if np.linalg.norm(x) == 0:
			x = np.array([1.0, 0.0, 0.0])
		x = x / np.linalg.norm(x)
		y = np.cross(z, x)

		matrix = np.eye(4)
		matrix[:3,:3] = np.c_[x,y,z]
		nodes.append(pyrender.Node(
		light=pyrender.DirectionalLight(color=c, intensity=intensity),
		matrix=matrix
		))

	return nodes


def clip(x1, y1, x2, y2):

    ret = np.clip([x1, y1, x2, y2], 0.0, 1.0)
    return ret[0], ret[1], ret[2], ret[3]

class chair_random(data_loader.base):

    name = 'chair'
    use = 'localization'

    def __init__(self, cfg, data='train', transform_data=None, target_transform=None):

        super(chair_random, self).__init__(cfg, transform_data, target_transform)
        
        #self.fname = "chair1.ply"
        #self.fname = "chair1_trans.ply"
        self.fname = "./dataset/chair1_trans.ply"
        self.max_detections = 4
        self.num_objects = 1
        self.initialize_scene()
        self.data = data
        if data == 'train':
            self.num_data = 2**10

        elif data == 'val':
            self.num_data = 2**8

        elif data == 'test':
            self.num_data = 2**8
            

        elif data == 'check':
            self.num_data = 128
            
        #self.indeces = self.get_index_all()
        self.sampler_R = None
        self.sampler_T = None
        self._loop = 0
    
    def initialize_dataset(self):
        self.indeces = self.get_index_all()
        self.sample_conditions_some()

    def sample_conditions(self):
        self._R = self.sampler_R.get(self.num_data)
        self._T = self.sampler_T.get(self.num_data)

    def sample_conditions_some(self):
        self._R, self._T, self._Obj= [], [], []
        for _ in range(self.num_data):
            randn = np.random.randint(1, self.num_objects+1)
            r = self.sampler_R.get(randn)
            t = self.sampler_T.get(randn)
            o = np.random.randint(0, self.num_objects, randn)

            self._R.append(r)
            self._T.append(t)
            self._Obj.append(o)

    def initialize_scene(self):

        bg_color = np.array([0.1, 0.1, 0.1])
        _trimesh = trimesh.load(self.fname)
        self.get_bbox3d(_trimesh)
        mesh = pyrender.Mesh.from_trimesh(_trimesh)

        intrinsicMat, camera = self.load_camera()
        self.intrinsicMat = intrinsicMat
        pose = [[ 1.0,  0.0,  0.0,  0.0],\
                [ 0.0, -1.0,  0.0,  0.0],\
                [ 0.0,  0.0, -1.0,  0.0],\
                [ 0.0,  0.0,  0.0,  1.0]]
        self.cam_pose_init = np.array(pose)

        light_color = np.array([1.0, 1.0, 1.0])
        light = pyrender.SpotLight(color=light_color, intensity=80.0,\
                                   innerConeAngle=np.pi/2.2,\
                                   outerConeAngle=np.pi/2.0)
        light_pose = [[1.0, 0.0, 0.0, 0.0],\
                      [0.0, 0.0, -1.0, -3.0],\
                      [0.0, 1.0, 0.0, -0.2],\
                      [0.0, 0.0, 0.0, 1.0]]
        self.light_pose = np.array(light_pose)
        self.mesh_pose = np.eye(4)
        scene = pyrender.Scene(nodes=None, bg_color=bg_color)
        scene.add(mesh, pose=self.mesh_pose)
        scene.add(camera, pose=self.cam_pose_init)
        scene.add(light, pose=self.light_pose)
        
        self.renderer = pyrender.OffscreenRenderer(1280, 720)
        self.scene = scene
        self.mesh = mesh

    def __len__(self):
        return self.num_data // self.batchsize

    def load_camera(self):
        fx = 504.53
        fy = 504.53
        cx = 640.
        cy = 360.
        cam = pyrender.camera.IntrinsicsCamera(fx, fy, cx, cy)

        intrinsicMat = [[fx,0,cx],[0,fy,cy],[0,0,1]]
        intrinsicMat = np.asanyarray(intrinsicMat)
        return intrinsicMat, cam


    def get_next0(self):
        if self._loop >= len(self.indeces):
            self._loop = 0
            raise StopIteration()
        
        indeces_local = self.indeces[self._loop]
        _color, _depth = self.get_images_RT(indeces_local)
        #_target = self.get_target(indeces_local, None)
        _target = self.get_target(indeces_local, (_color.shape[2], _color.shape[1]))

        self._loop += 1
        return _color, _target

    def get_next1(self):

        if self._loop >= len(self.indeces):
            self._loop = 0
            raise StopIteration()
        
        indeces_local = self.indeces[self._loop]
        _color, _depth = self.get_images_RT(indeces_local)
        _target = self.get_target(indeces_local, None)
        #_target = self.get_target(indeces_local, (_color.shape[2], _color.shape[1]))

        self._loop += 1
        return [[_color, _depth], _target]

    def __next__(self):
        return self.get_next0()

    def get_target(self, indeces_local, size = None):

        bbox = self.get_bbox2d_list(indeces_local, size)

        return bbox

    def get_image_RT(self, ind):
        _R = self._R[ind]
        _T = self._T[ind]

        for _r, _t in zip(_R, _T):
            #print(_r.shape, _t.shape)
            new_mat = np.eye(4)
            new_mat[0:3, 0:3] = _r
            new_mat[0, 3] = _t[0]
            new_mat[1, 3] = _t[1]
            new_mat[2, 3] = _t[2]
            
            #
            # full out
            #
            list(self.scene.mesh_nodes)[0].matrix = np.dot(new_mat, self.mesh_pose)
            color, depth = self.renderer.render(self.scene)
            break
        
        return color, depth


    def get_images_RT(self, indeces):
        
        colors, depths = [], []
        #for _r, _t in zip(_R, _T):
        for ind in indeces:
            color, depth = self.get_image_RT(ind)
            colors.append(color.tolist())
            depths.append(depth.tolist())

        colors = np.array(colors)
        depths = np.array(depths)
        #print(np.min(colors), np.max(colors))
        return colors, depths

    def get_bbox2d(self, _r, _t, size = None):

        new_mat = np.eye(4)
        new_mat[0:3, 0:3] = _r
        new_mat[0, 3] = _t[0]
        new_mat[1, 3] = _t[1]
        new_mat[2, 3] = _t[2]
        extrinsicMat = np.dot(new_mat, self.mesh_pose)
        coord2Dx = []
        coord2Dy = []
        for coord in self.coords3D:
            extMat = np.dot(extrinsicMat[:-1], coord)
            for mat in extMat:
                mat[0] = mat[0]/extMat[-1][0]
            fMat = np.dot(self.intrinsicMat, extMat)
            for mat in fMat:
                mat[0] = mat[0]/fMat[-1][0]
            u = fMat[0][0]
            v = fMat[1][0]
            coord2Dx.append(u)
            coord2Dy.append(v)
        codx = np.asanyarray(coord2Dx)
        cody = np.asanyarray(coord2Dy)

        x1 = min(codx)
        x2 = max(codx)
        y1 = min(cody)
        y2 = max(cody)
        
        #bbox2d = [min(codx), max(codx), max(cody),min(cody)]# ahmed
        if size is None:
        #if self.normalized == False:
            bbox2d = [x1, y1, x2, y2]
        else:
            # normalized coordinate
            #print(x1, y1, x2, y2)
            x1 = x1 / (size[0] - 1)
            y1 = y1 / (size[1] - 1)
            x2 = x2 / (size[0] - 1)
            y2 = y2 / (size[1] - 1)

            ret = np.clip([x1, y1, x2, y2], 0.0, 1.0)
            x1_s, y1_s, x2_s, y2_s = ret[0], ret[1], ret[2], ret[3]

            #print(x1, y1, x2, y2)
            bbox2d = [x1_s, y1_s, x2_s, y2_s]
        return bbox2d


    def get_bbox2d_list(self, indeces, size = None):

        bbox2d_list = []
        #print(len(self._R), len(self._T), len(self._R[0]), len(self._T[0]))
        for ind in indeces:
            bboxes = []
            for r, t, o in zip(self._R[ind], self._T[ind], self._Obj[ind]):
                bbox2d = self.get_bbox2d(r, t, size)
                bboxes.append(bbox2d + [o])
            #print(len(bboxes))
            bbox2d_list.append(bboxes)
        return bbox2d_list


    def get_bbox3d(self, tmesh):
        bbox3D = tmesh.bounding_box
        corners = trimesh.bounds.corners(bbox3D.bounds)
        #print('corners', corners)
        #print(corners)
        self.corners3D = corners
        coords3D = []
        for corner in corners:
            ro = np.reshape(corner, (-1,1))
            ro = np.vstack([ro,[1]])
            coords3D.append(ro)
        self.coords3D = np.asanyarray(coords3D)


    def get_colors_numpy(self, _target):
        colors, depths = self.get_images_numpy(_target)
        return colors


    def transform_batch(self, batch_idx, device):
        
        data, target = self.get_colors_numpy()
        return data, target



class chair_randomT(chair_random):
    
    def __init__(self, cfg, data='train', transform_data=None, target_transform=None):
        cfg_T, cfg_R = {}, {}
        cfg_R['rx'] = 0.0
        cfg_R['ry'] = 0.0
        cfg_R['rz'] = 0.0

        cfg_T['x_near'] = [-0.45, 0.45]
        cfg_T['x_far'] = [-3.0, 3.0]
        cfg_T['y_near'] = [-0.20, 0.20]
        cfg_T['y_far'] = [-0.70, 0.70]
        #cfg_T['z'] = [-6.0, -1.0]
        #cfg_T['z'] = [0.0, 3.5]
        cfg_T['z'] = [0.3, 3.5]

        self.sampler_T = spacial_sampling.T_perspective(cfg_T)
        self.sampler_R = spacial_sampling.R_const(cfg_R)

        super(chair_randomT, self).__init__(cfg, data, transform_data, target_transform)


class chair_randomR(chair_random):
    def __init__(self, cfg, data='train', transform_data=None, target_transform=None):

        super(chair_randomR, self).__init__(cfg, data, transform_data, target_transform)
        cfg_T, cfg_R = {}, {}
        cfg_T['x'] = 0.0
        cfg_T['y'] = 0.0
        #cfg_T['z'] = -1.0
        cfg_T['z'] = 1.0
        
        #cfg_R['rx'] = [np.deg2rad(160.0), np.deg2rad(200.0)]
        #cfg_R['ry'] = [np.deg2rad(-70.0), np.deg2rad(70.0)]
        #cfg_R['rz'] = [np.deg2rad(80.0), np.deg2rad(100.0)]
        #cfg_R['rz'] = [np.deg2rad(-10.0), np.deg2rad(10.0)]
        cfg_R['rx'] = [np.deg2rad(-10.0), np.deg2rad(10.0)]
        cfg_R['ry'] = [np.deg2rad(-40.0), np.deg2rad(40.0)]
        cfg_R['rz'] = [np.deg2rad(-10.0), np.deg2rad(10.0)]

        #cfg_R['rx'] = None
        #cfg_R['ry'] = None
        #cfg_R['rz'] = None

        self.sampler_T = spacial_sampling.T_const(cfg_T)
        self.sampler_R = spacial_sampling.R_random(cfg_R)


class chair_randomRT(chair_random):
    def __init__(self, cfg, data='train', transform_data=None, target_transform=None):
        super(chair_randomRT, self).__init__(cfg, data, transform_data, target_transform)
        cfg_T, cfg_R = {}, {}
        if True:
            cfg_T['x_near'] = [-0.5, 0.5]
            cfg_T['x_far'] = [-4.5, 4.5]
            cfg_T['y_near'] = [0.1, 0.7]
            cfg_T['y_far'] = [-0.9, 0.9]
            cfg_T['z'] = [0.5, 3.6]
        else:
            cfg_T['x_near'] = [-0.5, 0.5]
            cfg_T['x_far'] = [-5.5, 5.5]
            cfg_T['y_near'] = [0.1, 0.7]
            cfg_T['y_far'] = [-0.9, 0.9]
            cfg_T['z'] = [0.1, 3.6]
        
        #cfg_R['rx'] = [np.deg2rad(-10.0), np.deg2rad(10.0)]
        cfg_R['rx'] = [np.deg2rad(-30.0), np.deg2rad(30.0)]
        cfg_R['ry'] = [np.deg2rad(-30.0), np.deg2rad(30.0)]
        cfg_R['rz'] = [np.deg2rad(-10.0), np.deg2rad(10.0)]
        self.sampler_T = spacial_sampling.T_perspective(cfg_T)
        self.sampler_R = spacial_sampling.R_random(cfg_R)

        
class chair_constRT(chair_random):
    
    def __init__(self, cfg, data='train', transform_data=None, target_transform=None):
        
        super(chair_constRT, self).__init__(cfg, data, transform_data, target_transform)
        cfg_T, cfg_R = {}, {}
        cfg_R['rx'] = 30.0
        cfg_R['ry'] = 0.0
        cfg_R['rz'] = 0.0
        cfg_T['x'] = 0.0
        cfg_T['y'] = 0.0
        cfg_T['z'] = 0.0

        self.sampler_R = spacial_sampling.R_const(cfg_R)
        self.sampler_T = spacial_sampling.T_const(cfg_T)
        

def imshow_simple(cfg):
    data = chair_randomT(cfg, 'train')
    color, depth, target = data.get_images_numpy()
    import pylab as plt
    plt.figure()
    plt.subplot(121)
    plt.imshow(color)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(depth)
    plt.colorbar()
    plt.show()



def test1(cfg):
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl

    data = chair_randomRT(cfg, 'check')
    data.initialize_dataset()
    [[color, depth], target] = data.get_next1()
    print(np.min(color), np.max(color))

    color_0, target_0 = data.get_next0()
    print(np.min(color_0), np.max(color_0))
    #print("target_0", target_0)

    path = 'temp/'
    util.remove_files(path)
    util.make_directory(path)

    n_file = 0

    print(len(target), len(target[0]))

    plt.figure()
    for c, d, tt in zip(color, depth, target):

        n_file += 1
        fname = path + (5 - len(str(n_file))) * '0' + str(n_file)
        print(fname)
        cc = c
        dd = d

        plt.clf()
        plt.subplot(121)
        plt.imshow(cc)
        plt.colorbar()
        plt.subplot(122)

        print(len(tt))
        for t in tt:
            #t = tt[0]
            pX = [t[0], t[2], t[2], t[0], t[0]]
            pY = [t[3], t[3], t[1], t[1], t[3]]
            plt.plot(pX,pY)

        plt.imshow(dd)
        plt.colorbar()
        plt.savefig(fname)



def test2(cfg):

    data = chair_randomRT(cfg, 'train')




if __name__ == '__main__':
    
    print("start")

    np.random.seed(1234)
    
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.PATH = './data/'
    cfg.BATCHSIZE = 30

    test1(cfg)
    #test2(cfg)

    print("end")

