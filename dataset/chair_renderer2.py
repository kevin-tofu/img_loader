
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import trimesh
import pyrender
from pyrender.material import MetallicRoughnessMaterial
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

os.environ["PYOPENGL_PLATFORM"] = "egl"
from numpy import random

if __name__ == '__main__':
    import spacial_sampling, data_loader
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import util
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from dataset import spacial_sampling, data_loader
    import util

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
        self.objpath = cfg.PATH + cfg.object
        self.fnames = [self.objpath+f for f in listdir(self.objpath) if isfile(join(self.objpath, f))]
        self.max_detections = 4
        self.overlapLimit = cfg.overlapLimit
        self.num_objects = cfg.totalObjects
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

    def get_obj_sequence(self):
        nMeshes = len(self.meshes)
        perMesh =int(self.num_objects / nMeshes)
        extraMeshes = self.num_objects - np.multiply(perMesh, nMeshes)
        elements = list(range(len(self.meshes)))
        seq = []
        for e in elements:
            seq += list(np.full(shape=perMesh, fill_value=e, dtype=np.int))

        for i in range(extraMeshes):
            seq.append(random.choice(elements))
        random.shuffle(seq)
        self.meshSeq = seq
    
    def initialize_scene(self):

        self.bg_color = np.array([0 , 0, 0])
        self.trimeshes, self.meshes = [], []
        for obj in self.fnames:
            tmesh = trimesh.load(obj)
            tmesh.visual.vertex_colors = tmesh.visual.vertex_colors
            self.trimeshes.append(tmesh)
        for mesh in self.trimeshes:
            self.meshes.append(pyrender.Mesh.from_trimesh(mesh))
        self.get_obj_sequence()

        self.intrinsicMat, self.camera = self.load_camera()
        pose = [[ 1.0,  0.0,  0.0,  0.0],\
                [ 0.0, -1.0,  0.0,  0.0],\
                [ 0.0,  0.0, -1.0,  0.0],\
                [ 0.0,  0.0,  0.0,  1.0]]
        self.cam_pose_init = np.array(pose)
        light_color = np.array([1.0, 1.0, 1.0])
        self.light = pyrender.SpotLight(color=light_color, intensity=80.0,\
                                   innerConeAngle=np.pi/2.2,\
                                   outerConeAngle=np.pi/2.0)
        light_pose = [[1.0, 0.0, 0.0, 0.0],\
                      [0.0, 0.0, -1.0, -3.0],\
                      [0.0, 1.0, 0.0, -0.2],\
                      [0.0, 0.0, 0.0, 1.0]]
        self.light_pose = np.array(light_pose)
        self.mesh_pose = np.eye(4)
        self.renderer = pyrender.OffscreenRenderer(1280, 720)

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

    def get_bgimages(self, cfg):
        bgpath = cfg.PATH+cfg.bg
        bgfiles =[f for f in listdir(bgpath) if isfile(join(bgpath, f))]
        bgimages = []
        for f in bgfiles:
            if(f[-4:] == '.png'):
                bgimages.append(f)
        return bgimages

    def combine_obj_with_bg(self, cfg, color):
        bgimages = self.get_bgimages(cfg)
        newImgs = []
        for c in color:
            imgs = []
            for cx in c:
                cc = makeTransparent(np.asarray(cx))
                cc = Image.fromarray(cc.astype(np.uint8))

                randn = np.random.randint(0, len(bgimages))
                bgimg = Image.open(cfg.PATH+cfg.bg+bgimages[randn])
                bgimg = bgimg.resize((1280, 720), Image.ANTIALIAS).convert('RGBA')
                cimg = Image.alpha_composite(bgimg, cc)
                imgs.append(np.array(cimg))
            newImgs.append(imgs)
        return np.asarray(newImgs)

    def get_next0(self):
        if self._loop >= len(self.indeces):
            self._loop = 0
            raise StopIteration()
        
        indeces_local = self.indeces[self._loop]
        _color, _depth, _target = self.get_images_RT(indeces_local)
        for targ in _target:
            for coords in targ:
                for i in range(len(coords[:4])):
                    if(i%2 == 0):
                        coords[i] /= _color.shape[3]
                    else:
                        coords[i] /= _color.shape[2]

        self._loop += 1

        _color = self.combine_obj_with_bg(cfg, _color)
        return _color, _target

    def get_next1(self, cfg):

        if self._loop >= len(self.indeces):
            self._loop = 0
            raise StopIteration()
        
        indeces_local = self.indeces[self._loop]
        _color, _depth, _target = self.get_images_RT(indeces_local)
        self._loop += 1

        _color = self.combine_obj_with_bg(cfg, _color)
        return [[_color, _depth], _target]

    def __next__(self):
        return self.get_next0()

    def get_target(self, indeces_local, size = None):

        bbox = self.get_bbox2d_list(indeces_local, size)

        return bbox
    
    def get_target_seg(self, rtMat, nthMesh):

        s = pyrender.Scene(nodes=None, bg_color=self.bg_color)
        s.add(nthMesh, pose=rtMat)
        s.add(self.camera, pose=self.cam_pose_init)
        s.add(self.light, pose=self.light_pose)
        _, dx = self.renderer.render(s)
        dx /= np.asarray(dx).max()/255.0
        col = Image.fromarray(dx)
        gray = col.convert('L')
        bw = gray.point(lambda x: 0 if x==0 else 255, '1')
        im = np.asarray(bw)
        colors = set(np.unique(im))
        colors.remove(0)
        for color in colors:
            py, px = np.where(im == color)
            x1 = min(px)
            x2 = max(px)
            y1 = min(py)
            y2 = max(py)
            t = [x1, y1, x2, y2]
        return t, im
    
    def remove_unwanted_targets(self):
        overlapMat = []
        bi = self.binaryImages
        ts = self.sceneTargets
        for i in bi:
            o_counts = []
            for j in bi:
                if not((i==j).all()):
                    _, tpj = np.unique(j, return_counts=True)[-1]
                    _, tpi = np.unique(i, return_counts=True)[-1]
                    if(tpi < tpj):
                        sum=0
                        for x in range(i.shape[0]):
                            sum+=np.sum((j[x,:]!=0) & (i[x,:]!=0))
                        overlap = sum/tpi
                        if(overlap > self.overlapLimit):
                            o_counts.append(1)
                        else:
                            o_counts.append(0)
                    else:
                        o_counts.append(0)
                else:
                    o_counts.append(0)
            overlapMat.append(o_counts)
        fimg =[]
        for n in range(len(overlapMat)):
            if 1 in overlapMat[n]:
                fimg.append(n)
        ts = [i for j, i in enumerate(ts) if j not in fimg]
        self.sceneTargets = ts
    
    
    def get_image_RT(self, ind):
        _R = self._R[ind]
        _T = self._T[ind]

        cs, ds, ts, bi = [], [], [], []
        scene = pyrender.Scene(nodes=None, bg_color=self.bg_color)
        scene.add(self.camera, pose=self.cam_pose_init)
        scene.add(self.light, pose=self.light_pose)
        counter = 0
        for _r, _t in zip(_R, _T):
            new_mat = np.eye(4)
            new_mat[0:3, 0:3] = _r
            new_mat[0, 3] = _t[0]
            new_mat[1, 3] = _t[1]
            new_mat[2, 3] = _t[2]
            meshNum = self.meshSeq[counter]
            nthMesh = self.meshes[meshNum]
            scene.add(nthMesh, pose=new_mat)
            counter +=1
            bbox2d_seg, bnimg = self.get_target_seg(new_mat, nthMesh)
            bbox2d_seg.append(meshNum)
            ts.append(bbox2d_seg)
            bi.append(bnimg)
        self.sceneTargets = ts    
        self.binaryImages = bi
        self.remove_unwanted_targets()
        color, depth = self.renderer.render(scene)
        cs.append(color.tolist())
        ds.append(depth.tolist())
        ts = self.sceneTargets
        return cs, ds, ts


    def get_images_RT(self, indeces):
        
        colors, depths, targets = [], [], []
        for ind in indeces:
            color, depth, target = self.get_image_RT(ind)
            colors.append(color)
            depths.append(depth)
            targets.append(target)
        colors = np.array(colors)
        depths = np.array(depths)
        return colors, depths, targets

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
        
        if size is None:
            bbox2d = [x1, y1, x2, y2]
        else:
            x1 = x1 / (size[0] - 1)
            y1 = y1 / (size[1] - 1)
            x2 = x2 / (size[0] - 1)
            y2 = y2 / (size[1] - 1)

            ret = np.clip([x1, y1, x2, y2], 0.0, 1.0)
            x1_s, y1_s, x2_s, y2_s = ret[0], ret[1], ret[2], ret[3]
            bbox2d = [x1_s, y1_s, x2_s, y2_s]
        return bbox2d


    def get_bbox2d_list(self, indeces, size = None):

        bbox2d_list = []
        for ind in indeces:
            bboxes = []
            for r, t, o in zip(self._R[ind], self._T[ind], self._Obj[ind]):
                bbox2d = self.get_bbox2d(r, t, size)
                bboxes.append(bbox2d + [o])
            bbox2d_list.append(bboxes)
        return bbox2d_list


    def get_bbox3d(self, tmesh):
        bbox3D = tmesh.bounding_box
        corners = trimesh.bounds.corners(bbox3D.bounds)
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
        cfg_T['z'] = 1.0
        
        cfg_R['rx'] = [np.deg2rad(-10.0), np.deg2rad(10.0)]
        cfg_R['ry'] = [np.deg2rad(-40.0), np.deg2rad(40.0)]
        cfg_R['rz'] = [np.deg2rad(-10.0), np.deg2rad(10.0)]

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


def makeTransparent(image):
    ncc = []
    for i in image:
        cx = []
        for j in i:
            if(np.all((j == 0))):
                j = j.tolist()
                j.append(0)
                cx.append(j)
            else:
                j = j.tolist()
                j.append(255)
                cx.append(j)
        ncc.append(cx)
    image = np.asanyarray(ncc)
    return image



def test1(cfg):
    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl
    import matplotlib.image as mpimg
    data = chair_randomRT(cfg, 'check')
    data.initialize_dataset()
    [[color, depth], target] = data.get_next1(cfg)

    path = cfg.outPath
    util.remove_files(path)
    util.make_directory(path)
    n_file = 0
    
    plt.figure(figsize=(12, 6))
    for c, d, tt in zip(color, depth, target):
        
        n_file += 1
        fname = path + (5 - len(str(n_file))) * '0' + str(n_file)
        print(fname[-5:])
        
        plt.clf()
        plt.subplot(121)
        
        for cc in c:
            plt.imshow(cc)
            
        for t in tt:
            pX = [t[0], t[2], t[2], t[0], t[0]]
            pY = [t[3], t[3], t[1], t[1], t[3]]
            plt.plot(pX,pY)
        plt.colorbar()
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        plt.subplot(122)
        for t in tt:
            pX = [t[0], t[2], t[2], t[0], t[0]]
            pY = [t[3], t[3], t[1], t[1], t[3]]
            plt.plot(pX,pY)
        for dx in d:
            
            dd = dx
            plt.imshow(dd)
        plt.colorbar()
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig(fname)



def test2(cfg):

    data = chair_randomRT(cfg, 'train')




if __name__ == '__main__':
    
    print("start")

    np.random.seed(1234)
    
    from easydict import EasyDict as edict
    cfg = edict()
    cfg.PATH = '../../data/ProjectTokyo/dataset'
    cfg.object = '/ply_chairs/'
    cfg.bg = '/diode/val/indoors/scene_00019/scan_00183/'
    cfg.outPath =  '../../data/ProjectTokyo/share/chair_images/'
    cfg.totalObjects = 7
    cfg.overlapLimit = 0.6
    cfg.BATCHSIZE = 55

    test1(cfg)

    print("end")

