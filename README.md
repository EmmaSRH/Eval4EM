# Eval4EM
------------
This project is used for the evalation for EM 2D and 3D image segmentation task.

## Installation
------------

You can use one of the following two commands to install the required packages:
```
conda install --yes --file requirements.txt
pip install -r requirements.txt
```


## Evaluate for Cremi Data
------------

For each of the challenge categories, you find an evaluation class in
`cremi.evaluation`, which are `NeuronIds`, `Clefts`, and `SynapticPartners`.

After you read a test file `test` and a ground truth file `truth`, you can
evaluate your results by instantiating these classes as follows:
```python
from cremi.evaluation import NeuronIds, Clefts, SynapticPartners

neuron_ids_evaluation = NeuronIds(truth.read_neuron_ids())
(voi_split, voi_merge) = neuron_ids_evaluation.voi(test.read_neuron_ids())
adapted_rand = neuron_ids_evaluation.adapted_rand(test.read_neuron_ids())

clefts_evaluation = Clefts(test.read_clefts(), truth.read_clefts())
fp_count = clefts_evaluation.count_false_positives()
fn_count = clefts_evaluation.count_false_negatives()
fp_stats = clefts_evaluation.acc_false_positives()
fn_stats = clefts_evaluation.acc_false_negatives()

synaptic_partners_evaluation = SynapticPartners()
fscore = synaptic_partners_evaluation.fscore(
    test.read_annotations(),
    truth.read_annotations(),
    truth.read_neuron_ids())
```


## Evaluate For Binary Segmentation Task
------------

We provide Dice coefficient, Jaccard coefficient, Precision, Precision, F1 score.

Run the following command to use the tool:
```
python eval_binary_seg.py --gt 'path to gt mask' --pre 'path to pre mask' --save_file 'results/evaluation_results.txt'
```

The following steps will be executed by the script:
1) Load the following 3D arrays:
- GT segmentation volume
- prediction segmentation volume

2) Calculate the Dice coefficient, Jaccard coefficient, Precision, Precision, F1 score

3) Save the results to txt file


## Evaluate for Instance Segmentation
------------
Run the following command to use the tool:
```
python eval_instance_seg.py -gt demo_data/lucchi_gt_test.h5 -p demo_data/lucchi_pred_UNet_label_test.h5 -ph demo_data/lucchi_pred_UNet_heatmap_test.h5
```

The following steps will be executed by the script:
1) Load the following 3D arrays:
- GT segmentation volume
- prediction segmentation volume
- model prediction matrix / scores matrix in order to get the prediciton score of each voxel

2) Create the necessary tables to compute the mAP:
- iou_p.txt contains the different prediction ids, the prediction scores, and their matched ground trught (gt) ids. Each prediciton is matched with gt ids from 4 different size ranges (based on number of instance voxels). Each of these ranges contains the matched  gt id, its size and the intersection over union (iou) score. 
- iou_fn.txt contains false negatives, as well as instances that have been matched with a worse iou than another instance.  

3) Evaluate the model performance with mAP by using the 3D optimized evaluation script  and the 2 tables mentioned above.


## Evaluate For Denoise and Interplation Tasks
------------

We provide PSNR, SSIM, LPIPS, Foutier Ring Correlation scores.

Run the following command to use the tool:
```
python eval_binary_seg.py --gt 'path to gt mask' --pre 'path to pre mask' --save_file 'results/evaluation_results.txt'
```

The following steps will be executed by the script:
1) Load the following 3D arrays:
- GT segmentation volume
- prediction segmentation volume

2) Calculate the scores

3) Save the results to txt file

Acknowledgements
----------------

Evaluation code contributed by:

  * [Jan Funke](https://github.com/funkey)
  * [Juan Nunez-Iglesias](http://github.com/jni)
  * [Philipp Hanslovsky](http://github.com/hanslovsky)
  * [Stephan Saalfeld](http://github.com/axtimwalde)

