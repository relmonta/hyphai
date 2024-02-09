
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset
import hyphai.utils as utils


class CloudDataset(Dataset):
    """Cloud type images dataset.

    """

    def __init__(self, list_IDs, root_dir, transform=None, dim=(256, 256), context_size=2,
                 n_classes=12, leadtime=6, levels: dict = None):
        """__init__ method

        Args:
            list_IDs (list): Data indices
            root_dir (str, optional): data directory.
            transform (callable, optional): Optional transform to be applied. Defaults to None.
            dim (tuple, optional): Image resolution. Defaults to (256, 256).
            context_size (int, optional): Context observations size. Defaults to 2 (at t=0 and t=-1).
            n_classes (int, optional): Number of classes. Defaults to 12.
            leadtime (int, optional): Prediction lead-time. Defaults to 6.
            levels (dict, optional): classification levels. Defaults to None.
        """
        self.root_dir = root_dir
        self.list_IDs = list_IDs
        self.dim = dim
        self.context_size = context_size
        self.n_classes = n_classes
        self.leadtime = leadtime
        self.levels = levels
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns:
            int: Size of data
        """
        return len(self.list_IDs)

    def __getitem__(self, idx) -> dict:

        # Initialization
        y = []
        for k in range(self.leadtime):
            y += [np.empty(self.dim)]

        # Store inputs
        sample = np.load(self.root_dir + str(self.list_IDs[idx]) + '.npy')
        sample = utils.cloud_levels(sample, self.levels)
        # U-net input as shape (C,H,W)
        x_context = np.moveaxis(sample[:, :, :self.context_size].copy(), -1, 0)
        # Initial state as (H,W, n_classes)
        X_0 = utils.one_hot(
            sample[:, :, self.context_size - 1], num_classes=self.n_classes)

        # Store targets
        for j in range(self.leadtime):
            y[j] = sample[:, :, self.context_size + j].copy()

        sample_ = {'X': [x_context, X_0], 'y': y}
        if self.transform:
            sample_ = self.transform(sample_)

        return sample_


class Rotate(object):
    """Rotate randomly the images in a sample.

    """

    def __init__(self) -> None:
        pass

    def __call__(self, sample) -> dict:
        random_rot = np.random.randint(0, 4)

        x_context = sample['X'][0]
        X_0 = sample['X'][1]
        y = sample['y']

        for i in range(x_context.shape[0]):
            x_context[i, ...] = np.rot90(x_context[i, ...], random_rot)

        for i in range(X_0.shape[-1]):
            X_0[..., i] = np.rot90(X_0[..., i], random_rot)
        # outputs
        for i in range(len(y)):
            y[i] = np.rot90(y[i], random_rot)

        return {'X': [x_context, X_0], 'y': y}


class ToTensor(object):
    """Convert convert samples from Numpy arrays to torch Tensors.

    """

    def __init__(self, target_one_hot=False) -> None:
        self.target_one_hot = target_one_hot

    def __call__(self, sample) -> torch.Tensor:
        x_context = sample['X'][0]
        X_0 = sample['X'][1]
        y = sample['y']
        y_torch = []

        for i in range(len(y)):
            if self.target_one_hot:
                y_torch += [torch.from_numpy(utils.one_hot(
                    y[i], num_classes=X_0.shape[-1], axis=0)).float()]
            else:
                y_torch += [torch.from_numpy(y[i].copy()).long()]

        return {'X': [torch.from_numpy(x_context).float(),
                      torch.from_numpy(X_0).unsqueeze(0).float()],
                'y': y_torch}