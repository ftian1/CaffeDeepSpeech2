# --------------------------------------------------------
# Deep Speech 2 Caffe Implementation
# Written by Tian, Feng <feng.tian@intel.com>
# --------------------------------------------------------

"""The data layer used during training to train a DS2 network.

DS2DataLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import json
from audio_data_loader import SpectrogramDataset

class DS2DataLayer(caffe.Layer):
    """DeepSpeech2 data layer used for training."""

    def setup(self, bottom, top):
        """Setup the DS2DataLayer."""
	audio_conf = dict(sample_rate=16000,
			window_size=.02,
			window_stride=.01,
			window="hamming",
			noise_dir=None,
			noise_prob=0.4,
			noise_levels=(0.0, 0.5))
	train_manifest = "data/an4_train_manifest.csv"
	val_manifest = "data/an4_val_manifest.csv"
	with open("data/labels.json") as label_file:
	    labels = str(''.join(json.load(label_file)))
	self.train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=train_manifest, labels=labels,
        	                           normalize=True, augment=False)
	self.test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=val_manifest, labels=labels,
        	                          normalize=True, augment=False)
        
        # data blob: holds a batch of N images, each with 3 channels
	self._name_to_top_map = {}
        idx = 0
        top[idx].reshape(20, 1, 161, 81)
        self._name_to_top_map['inputs'] = idx
        idx += 1
	top[idx].reshape(20, 1, 100, 100)
	self._name_to_top_map['targets'] = idx
	idx += 1
	top[idx].reshape(1)
	self._name_to_top_map['input_percentages'] = idx
	idx += 1
	top[idx].reshape(1)
	self._name_to_top_map['target_sizes'] = idx

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
    	def func(p):
        	return p[0].shape[1]
        dataset = self.train_dataset
        idx = dataset.get_next_batches()
	batch = []
	for i in idx:
        	batch.append(dataset[i])
    	longest_sample = max(batch, key=func)[0]
   	freq_size = longest_sample.shape[0]
    	minibatch_size = len(batch)
    	max_seqlength = longest_sample.shape[1]
    	inputs = np.zeros((minibatch_size, 1, freq_size, max_seqlength))
    	input_percentages = [] 
    	target_sizes = []
    	targets = []
    	for x in range(minibatch_size):
        	sample = batch[x]
        	tensor = sample[0]
        	target = sample[1]
        	seq_length = tensor.shape[1]
		inputs[x][0][:, 0:seq_length] = tensor
        	input_percentages.append(seq_length / float(max_seqlength))
        	target_sizes.append(len(target))
        	targets.extend(target)
	blobs = {}
	blobs["inputs"] = inputs
	blobs["targets"] = np.asarray(targets, dtype=np.int)
    	blobs["input_percentages"] = np.array(input_percentages, dtype=np.float)
	blobs["target_sizes"] = np.array(target_sizes, dtype=np.int) 
	return blobs
