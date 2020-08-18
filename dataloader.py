import numpy as np
import skimage
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F
from io import BytesIO


def get_paths(fname,opt):
    paths = []
    with open(fname, 'r') as f:
        for line in f:
            if opt.system == 'Jarvis':
                temp = '/media/data/salman/'+str(line).strip()
            elif opt.system == 'FPM':
                temp = '/media/salman/'+str(line).strip()
            paths.append(temp)
    return paths

def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    try:
        magic = np.fromfile(f, np.float32, count=1)[0]    # For Python3.x
    except:
        magic = np.fromfile(f, np.float32, count=1)       # For Python2.x
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        #print("Reading %d x %d flo file" % (h, w))
        data2d = np.fromfile(f, np.float32, count=2 * w[0] * h[0])
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (h[0], w[0], 2))
    f.close()
    return torch.from_numpy(data2d)



class DatasetFromFilenames1ch:

    def __init__(self, filenames_loc,opt):
        self.filenames = filenames_loc
        self.paths = get_paths(self.filenames,opt)
        self.num_im = len(self.paths)
        self.totensor = torchvision.transforms.ToTensor()
        self.crop = torchvision.transforms.CenterCrop((opt.sceneH,opt.sceneW))
        self.opt = opt

    def __len__(self):
        return len(self.paths_orig)

    def __getitem__(self, index):
        # obtain the image paths
#         print(index)
        im_path = self.paths_orig[index % self.num_im]
        im1_path = im_path+'_img1.ppm'
        im2_path = im_path+'_img2.ppm'
        flo_path = im_path+'_flow.flo'
        # load images (grayscale for direct inference)
        im1 = Image.open(im1_path)
        im1 = im.convert('RGB')
        im2 = Image.open(im2_path)
        im2 = im.convert('RGB')
        im1 = self.crop(im1)
        im2 = self.crop(im2)
        # print(im.size)
        im1 = self.totensor(im1)[0,:,:].unsqueeze(0)
        im2 = self.totensor(im2)[0,:,:].unsqueeze(0)
        flo = read_flow(flo_path)
        flo = flo.permute(2,0,1)
        # im = (im-0.5)*2
        # print(im.shape)
        return im1,im2,flo