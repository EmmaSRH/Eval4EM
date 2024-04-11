# Eval4EM
------------
This project is used for the evalation for EM 2D and 3D image segmentation task.

## Installation
------------

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

```
python eval_binary_seg.py --gt 'path to gt mask' --pre 'path to pre mask' 
```


## Evaluate for hdf data same as Cremi data
------------

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
See the included `example_evaluate.py` for more details. The metrics are
described in more detail on the [CREMI Challenge website](http://cremi.org/metrics/).

Acknowledgements
----------------

Evaluation code contributed by:

  * [Jan Funke](https://github.com/funkey)
  * [Juan Nunez-Iglesias](http://github.com/jni)
  * [Philipp Hanslovsky](http://github.com/hanslovsky)
  * [Stephan Saalfeld](http://github.com/axtimwalde)

