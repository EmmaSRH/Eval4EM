import numpy as np
import h5py
import imageio
from scipy.ndimage import zoom
def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the dice coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

def jaccard_coefficient(mask_gt, mask_pred):
    """Compute jaccard coefficient.

    compute the jaccard coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      the jaccard coeffcient as float. If both masks are empty, the result is NaN
    """
    volume_intersect = (mask_gt & mask_pred).sum()
    volume_sum = mask_gt.sum() + mask_pred.sum() - volume_intersect
    if volume_sum == 0:
        return np.NaN
    jaccard = volume_intersect /volume_sum
    return jaccard
def precision_recall(gt, pred):
    tp = np.sum(gt * pred)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp
    if tp == 0 and fp == 0:
        precision = 0  # 防止分母为0
    else:
        precision = tp / (tp + fp)
    if tp == 0 and fn == 0:
        recall = 0  # 防止分母为0
    else:
        recall = tp / (tp + fn)
    return precision, recall

def f1_score(gt, pred):
    precision, recall = precision_recall(gt, pred)
    if precision + recall == 0:
        f1 = np.NaN  # 防止分母为0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return f1

def load_data(data_path,dataset = None):
    """
    Load volumetric data in HDF5, TIFF or PNG formats.
    """
    img_type = data_path.split('.')[-1]
    if img_type in ['h5', 'hdf5']:
        data = readh5(data_path, dataset)
    elif 'tif' in img_type:
        data = imageio.volread(data_path).squeeze()
        if data.ndim == 4:
            # convert (z,c,y,x) to (c,z,y,x) order
            data = data.transpose(1, 0, 2, 3)
    elif 'png' in img_type:
        data = imageio.imread(data_path)
        if data.ndim == 4:
            # convert (z,y,x,c) to (c,z,y,x) order
            data = data.transpose(3, 0, 1, 2)
    else:
        raise ValueError('unrecognizable file format for %s' % (data_path))
    assert data.ndim in [3, 4], "Currently supported volume data should " + \
                                "be 3D (z,y,x) or 4D (c,z,y,x), got {}D".format(data.ndim)
    return data


def readh5(filename, dataset=None):
    fid = h5py.File(filename, 'r')
    if dataset is None:
        # load the first dataset in the h5 file
        dataset = list(fid)[0]
    return np.array(fid[dataset])