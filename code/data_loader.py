import torch
from torch.utils.data import Dataset
import os
import random
import numpy as np
import rasterio
from tqdm import tqdm
from natsort import natsorted
import torchvision.transforms as tr

def get_window_offsets(img_dim, psize, stride):
    max_dim = (np.array(img_dim) // psize) * psize - psize

    ys = np.arange(0, img_dim[0], stride)
    xs = np.arange(0, img_dim[1], stride)

    tlc = np.array(np.meshgrid(ys, xs)).T.reshape(-1, 2)
    tlc = tlc[tlc[:, 0] <= max_dim[0]]
    tlc = tlc[tlc[:, 1] <= max_dim[1]]

    return tlc.astype(int)

class multiCD(Dataset):
    def __init__(self, root, split="all", fp_modifier=10, transform=None, s2_channel_type=3, run_on_onera_patches=False, use_pre_sliced=True, normalize=False):

        self.splits         = {"train": ['abudhabi', 'aguasclaras', 'beihai', 'beirut', 'bercy', 'bordeaux', 'cupertino', 'hongkong', 'mumbai', 'nantes', 'paris', 'pisa', 'rennes', 'saclay_e'],
                               "test": ['brasilia', 'chongqing', 'dubai', 'lasvegas', 'milano', 'montpellier', 'norcia', 'rio', 'saclay_w', 'valencia']}
        self.splits["all"]  = self.splits["train"] + self.splits["test"]

        self.split          = split
        self.names          = natsorted(self.splits[self.split])  # get list of names of ROI
        self.root_dir       = root
        self.run_on_onera   = run_on_onera_patches
        self.normalize      = normalize  # whether to z-score S1 & S2 or not, only do on whole images (not on patches)

        # settings pertaining to offline or online slicing
        self.use_pre_sliced = use_pre_sliced
        self.patch_indices  = []
        self.patch_size     = 96
        self.stride         = int(self.patch_size/2) - 1  # self.patch_size/2

        # 0-RGB | 1-RGBIr | 2-All bands s.t. resolution <= 20m | 3-All bands | 4-All bands Sentinel-2 & Sentinel-1 | 5-RGB bands Sentinel-2 & Sentinel-1

        if s2_channel_type in [0, 5]:
            self.s2_channels = 1 + np.arange(1, 4)
        elif s2_channel_type == 1:
            self.s2_channels = 1 + np.array([1, 2, 3, 7])
        elif s2_channel_type == 2:
            self.s2_channels = 1 + np.array([1, 2, 3, 4, 5, 6, 7, 8, 11, 12])
        elif s2_channel_type in [3, 4]:
            self.s2_channels = 1 + np.arange(0, 13)

        # keep track of changed pixels and total pixels in samples (of the change map labels)
        self.true_pix       = 0
        self.n_pix          = 0

        if not self.use_pre_sliced:
            # load full-scene images into memory
            self.full_scenes = dict()
            for roi in tqdm(self.names):
                # get the full-scene images, note: get_img_np_instead_of_tensor is already doing whole-image preprocessing
                self.full_scenes[roi] = self.get_img_np_instead_of_tensor(roi)
                self.true_pix += np.count_nonzero(self.full_scenes[roi]['label'])
                self.n_pix    += self.full_scenes[roi]['label'].size
                self.full_scenes[roi]['patches'] = get_window_offsets(self.full_scenes[roi]['time_1']['S2'].shape[1:], self.patch_size, self.stride)
            # pre-compute lookup tables of patches, to be used with online slicing
            self.patch_indices = np.cumsum([0]+[len(self.full_scenes[roi]['patches']) for roi in self.names])
            self.indices_dict  = {}
            for idx, index in enumerate(self.patch_indices): self.indices_dict[idx] = (self.names + ['end'])[idx]

        if transform:
            data_transform = tr.Compose([RandomFlip(), RandomRot()])
        else:
            data_transform = None

        # define patch transform for data augmentation
        self.transform      = data_transform
        self.paths          = [] if self.split=='test' or not self.use_pre_sliced else self.get_paths()  # get a list of paths
        self.n_samples      = len(self.paths) if self.use_pre_sliced else self.patch_indices[-1]

        # calculate a weighting of pixels
        self.weights = [] if self.split=='test' else [fp_modifier * 2 * self.true_pix / self.n_pix, 2 * (self.n_pix - self.true_pix) / self.n_pix]

    def get_paths(self):
        paths        = []
        modalities   = ["S1", "S2"]
        time_points  = [1, 2]
        s2_patch_dir = 'S2_patches_original' if self.run_on_onera else 'S2_patches'

        for roi in tqdm(self.splits[self.split]):
            path = os.path.join(self.root_dir, f"{modalities[0]}_patches", roi, f"imgs_{time_points[0]}")

            # get all files
            s1_1 = [os.path.join(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and ".tif" in f)]
            # sort via file names according to dates
            s1_1 = natsorted(s1_1)

            # get paired files and check for proper directory structure
            s1_2    = [f.replace('imgs_1','imgs_2') for f in s1_1 if os.path.isfile(f.replace('imgs_1','imgs_2'))]
            s2_1    = [f.replace('S1_patches', s2_patch_dir) for f in s1_1 if os.path.isfile(f.replace('S1_patches', s2_patch_dir))]
            s2_2    = [f.replace('imgs_1','imgs_2') for f in s2_1 if os.path.isfile(f.replace('imgs_1','imgs_2'))]
            label   = [f.replace('S1_patches','masks_patches').replace('imgs_1/','') for f in s1_1 if os.path.isfile(f.replace('S1_patches','masks_patches').replace('imgs_1/',''))]

            assert len(s1_1) == len(s1_2) == len(s2_1) == len(s2_2)  == len(label)

            for idx in range(len(s1_1)):
                sample = {'time_1': {'S1': s1_1[idx], 'S2': s2_1[idx]},
                          'time_2': {'S1': s1_2[idx], 'S2': s2_2[idx]},
                          'label': label[idx]}
                paths.append(sample)

                # keep track of number of changed and total pixels
                patch_pix = self.read_img(label[idx], [1]) - 1
                self.true_pix += np.sum(patch_pix)
                self.n_pix += np.prod(patch_pix.shape)
        return paths
    
    def read_img(self, path_IMG, bands):
        tif = rasterio.open(path_IMG)
        return tif.read(tuple(bands)).astype(np.float32)
    
    def rescale(self, img, oldMin, oldMax):
        oldRange = oldMax - oldMin
        img      = (img - oldMin) / oldRange
        return img

    def process_MS(self, img):
        intensity_min, intensity_max = 0, 10000                 # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)        # intensity clipping to a global unified MS intensity range
        img = self.rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches)
        return img

    def process_SAR(self, img):
        dB_min, dB_max = -25, 0                                 # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                      # intensity clipping to a global unified SAR dB range
        img = self.rescale(img, dB_min, dB_max)    
        return img

    def get_img(self, roi_name):
        # get path to full images
        containing_split = "Train" if roi_name in self.splits["train"] else "Test"
        containing_split = f"Onera Satellite Change Detection dataset - {containing_split} Labels"

        s1_1_path = os.path.join(self.root_dir, "S1", roi_name, "imgs_1", "transformed")
        s1_2_path = os.path.join(self.root_dir, "S1", roi_name, "imgs_2", "transformed")
        s1_1 =  torch.from_numpy(self.process_SAR(self.read_img(os.path.join(s1_1_path, os.listdir(s1_1_path)[0]), [1, 2])))
        s1_2 =  torch.from_numpy(self.process_SAR(self.read_img(os.path.join(s1_2_path, os.listdir(s1_2_path)[0]), [1, 2])))

        if self.run_on_onera:
            s2_1_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Images", roi_name, "imgs_1_rect")
            s2_2_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Images", roi_name, "imgs_2_rect")
            indices   = self.s2_channels - 1  # convert rasterio indices back to np indices
            s2_1 = torch.from_numpy(self.process_MS(np.array([self.read_img(os.path.join(s2_1_path, s2_1_file), [1]) for s2_1_file in natsorted(os.listdir(s2_1_path))])[indices, 0, ...]))
            s2_2 = torch.from_numpy(self.process_MS(np.array([self.read_img(os.path.join(s2_2_path, s2_2_file), [1]) for s2_2_file in natsorted(os.listdir(s2_2_path))])[indices, 0, ...]))
        else:
            s2_1_path = os.path.join(self.root_dir, "S2", roi_name, "imgs_1", "transformed")
            s2_2_path = os.path.join(self.root_dir, "S2", roi_name, "imgs_2", "transformed")
            s2_1 =  torch.from_numpy(self.process_MS(self.read_img(os.path.join(s2_1_path, os.listdir(s2_1_path)[0]), self.s2_channels)))
            s2_2 =  torch.from_numpy(self.process_MS(self.read_img(os.path.join(s2_2_path, os.listdir(s2_2_path)[0]), self.s2_channels)))
        if self.normalize:
            # z-standardize the whole images (after already doing preprocessing, this may matter for the non-linear SAR transforms)
            s2_1 = (s2_1 - s2_1.mean()) / s2_1.std()
            s2_2 = (s2_2 - s2_2.mean()) / s2_2.std()

        mask_path   = os.path.join(self.root_dir, containing_split, roi_name, "cm", f"{roi_name}-cm.tif")
        label       =  self.read_img(mask_path, [1])[0] - 1

        imgs = {'time_1': {'S1': s1_1, 'S2': s2_1},
                'time_2': {'S1': s1_2, 'S2': s2_2},
                'label': label}
        return imgs

    def get_img_np_instead_of_tensor(self, roi_name):
        # get path to full images

        containing_split = "Train" if roi_name in self.splits["train"] else "Test"
        containing_split = f"Onera Satellite Change Detection dataset - {containing_split} Labels"

        s1_1_path = os.path.join(self.root_dir, "S1", roi_name, "imgs_1", "transformed")
        s1_2_path = os.path.join(self.root_dir, "S1", roi_name, "imgs_2", "transformed")
        s1_1 =  self.process_SAR(self.read_img(os.path.join(s1_1_path, os.listdir(s1_1_path)[0]), [1, 2]))
        s1_2 =  self.process_SAR(self.read_img(os.path.join(s1_2_path, os.listdir(s1_2_path)[0]), [1, 2]))

        if self.run_on_onera:
            s2_1_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Images", roi_name, "imgs_1_rect")
            s2_2_path = os.path.join(self.root_dir, "Onera Satellite Change Detection dataset - Images", roi_name, "imgs_2_rect")
            indices   = self.s2_channels - 1  # convert rasterio indices back to np indices
            s2_1 = self.process_MS(np.array([self.read_img(os.path.join(s2_1_path, s2_1_file), [1]) for s2_1_file in natsorted(os.listdir(s2_1_path))])[indices, 0, ...])
            s2_2 = self.process_MS(np.array([self.read_img(os.path.join(s2_2_path, s2_2_file), [1]) for s2_2_file in natsorted(os.listdir(s2_2_path))])[indices, 0, ...])
        else:
            s2_1_path = os.path.join(self.root_dir, "S2", roi_name, "imgs_1", "transformed")
            s2_2_path = os.path.join(self.root_dir, "S2", roi_name, "imgs_2", "transformed")
            s2_1 =  self.process_MS(self.read_img(os.path.join(s2_1_path, os.listdir(s2_1_path)[0]), self.s2_channels))
            s2_2 =  self.process_MS(self.read_img(os.path.join(s2_2_path, os.listdir(s2_2_path)[0]), self.s2_channels))
        if self.normalize:
            # z-standardize the whole images (after already doing preprocessing, this may matter for the non-linear SAR transforms)
            s2_1 = (s2_1 - s2_1.mean()) / s2_1.std()
            s2_2 = (s2_2 - s2_2.mean()) / s2_2.std()

        mask_path   = os.path.join(self.root_dir, containing_split, roi_name, "cm", f"{roi_name}-cm.tif")
        label       =  self.read_img(mask_path, [1])[0] - 1

        imgs = {'time_1': {'S1': s1_1, 'S2': s2_1},
                'time_2': {'S1': s1_2, 'S2': s2_2},
                'label': label}
        return imgs

    def __getitem__(self, idx):

        if self.use_pre_sliced:
            # use patches sliced before (offline) by pre-processing script
            s1_1 = self.process_SAR(self.read_img(self.paths[idx]['time_1']['S1'], [1, 2]))
            s2_1 = self.process_MS(self.read_img(self.paths[idx]['time_1']['S2'], self.s2_channels))
            s1_2 = self.process_SAR(self.read_img(self.paths[idx]['time_2']['S1'], [1, 2]))
            s2_2 = self.process_MS(self.read_img(self.paths[idx]['time_2']['S2'], self.s2_channels))
            label= self.read_img(self.paths[idx]['label'], [1]) - 1

            sample = {'time_1': {'S1': s1_1, 'S2': s2_1},
                      'time_2': {'S1': s1_2, 'S2': s2_2},
                      'label': label,
                      'idx': idx,
                     }
        else:
            # load full-scene images and slice into patches online
            # (mirror-pad) and slice into patches
            first_idx = np.where(self.patch_indices>idx)[0][0]-1
            self.indices_dict[first_idx]
            self.full_scenes[self.indices_dict[first_idx]]['patches']
            # map the queried index to the current ROI's patch slice anchors (top left corner)
            patch_idx = self.full_scenes[self.indices_dict[first_idx]]['patches'][idx-self.patch_indices[first_idx]]
            # read the actual patch, given the full-scene image and the patch indices
            s1_1 = self.full_scenes[list(self.full_scenes.keys())[first_idx]]['time_1']['S1'][:, patch_idx[0]:(patch_idx[0]+self.patch_size), patch_idx[1]:(patch_idx[1]+self.patch_size)]
            s1_2 = self.full_scenes[list(self.full_scenes.keys())[first_idx]]['time_2']['S1'][:, patch_idx[0]:(patch_idx[0]+self.patch_size), patch_idx[1]:(patch_idx[1]+self.patch_size)]
            s2_1 = self.full_scenes[list(self.full_scenes.keys())[first_idx]]['time_1']['S2'][:, patch_idx[0]:(patch_idx[0]+self.patch_size), patch_idx[1]:(patch_idx[1]+self.patch_size)]
            s2_2 = self.full_scenes[list(self.full_scenes.keys())[first_idx]]['time_2']['S2'][:, patch_idx[0]:(patch_idx[0]+self.patch_size), patch_idx[1]:(patch_idx[1]+self.patch_size)]
            label= self.full_scenes[list(self.full_scenes.keys())[first_idx]]['label'][patch_idx[0]:(patch_idx[0]+self.patch_size), patch_idx[1]:(patch_idx[1]+self.patch_size)][None]

            # note: no preprocessing done here for online slicing as we already do preprocess when loading the full scene, on a whole-image basis
            sample = {'time_1': {'S1': s1_1, 'S2': s2_1},
                      'time_2': {'S1': s1_2, 'S2': s2_2},
                      'label': label,
                      'idx': idx,
                     }
        # apply data augmentation
        if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        # length of generated list
        return self.n_samples


class RandomFlip(object):
    """Flip randomly the images in a sample, right to left side."""

    def __call__(self, sample):
        I1, I2, I1_b, I2_b, label = sample['time_1']['S2'], sample['time_2']['S2'], sample['time_1']['S1'], sample['time_2']['S1'], sample['label']

        if random.random() > 0.5:
            I1 = I1[:, :, ::-1].copy()
            I2 = I2[:, :, ::-1].copy()
            I1_b = I1_b[:, :, ::-1].copy()
            I2_b = I2_b[:, :, ::-1].copy()
            label = label[:, :, ::-1].copy()
            sample = {'time_1': {'S1': I1_b, 'S2': I1},
                      'time_2': {'S1': I2_b, 'S2': I2},
                      'label': label,
                      'idx': sample['idx'],
                     }
        return sample


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    def __call__(self, sample):
        I1, I2, I1_b, I2_b, label = sample['time_1']['S2'], sample['time_2']['S2'], sample['time_1']['S1'], sample['time_2']['S1'], sample['label']

        n = random.randint(0, 3)
        if n:
            I1 = I1
            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
            I2 = I2
            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
            I1_b = I1_b
            I1_b = np.rot90(I1_b, n, axes=(1, 2)).copy()
            I2_b = I2_b
            I2_b = np.rot90(I2_b, n, axes=(1, 2)).copy()
            label = sample['label']
            label = np.rot90(label, n, axes=(1, 2)).copy()
            sample = {'time_1': {'S1': I1_b, 'S2': I1},
                      'time_2': {'S1': I2_b, 'S2': I2},
                      'label': label,
                      'idx': sample['idx'],
                     }
        return sample
