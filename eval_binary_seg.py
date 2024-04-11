import argparse
import pdb

from utils import compute_dice_coefficient
from utils import jaccard_coefficient
from utils import precision_recall
from utils import f1_score
from utils import load_data
import numpy as np


def write_to_txt(dice,jaccard,precision,recall,f1):
    output_file = "evaluation_results.txt"
    with open(output_file, "w") as f:
        f.write("Dice coefficient: {}\n".format(dice))
        f.write("Jaccard coefficient: {}\n".format(jaccard))
        f.write("Precision: {}\n".format(precision))
        f.write("Recall: {}\n".format(recall))
        f.write("F1 score: {}\n".format(f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str,
                        default='test_examples/gt.npy',
                        help='dir for ground truth volume data')
    parser.add_argument('--pre', type=str,
                        default='test_examples/gt.npy',
                        help='dir for prediction volume data')
    args = parser.parse_args()

    # gt_volume = load_data(args.gt)
    # pre_volume = load_data(args.pre)

    # Example usage
    # Assuming gt_volume and pred_volume are numpy arrays representing ground truth and prediction volumes
    gt_volume = np.array([[[0, 1, 1],
                          [1, 0, 1],
                          [1, 0, 0]]])
    print(gt_volume.shape)
    pdb.set_trace()

    pred_volume = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 0, 0]])

    dice = compute_dice_coefficient(gt_volume, pred_volume)
    jaccard = jaccard_coefficient(gt_volume, pred_volume)
    precision, recall = precision_recall(gt_volume, pred_volume)
    f1 = f1_score(gt_volume, pred_volume)

    print("Dice coefficient:", dice)
    print("Jaccard coefficient:", jaccard)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 score:", f1)

    write_to_txt(dice,jaccard,precision,recall,f1)