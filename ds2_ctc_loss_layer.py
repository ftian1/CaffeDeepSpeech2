# --------------------------------------------------------
# Deep Speech 2 Caffe Implementation
# Written by Tian, Feng <feng.tian@intel.com>
# --------------------------------------------------------

"""The data layer used during training to train a DS2 network.

DS2CtcLossLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np
import ctc

class DS2CtcLossLayer(caffe.Layer):
    """DeepSpeech2 transpose layer used for training."""

    def setup(self, bottom, top):
        """Setup the DS2CtcLossLayer."""
	assert len(bottom) == 4
	top[0].reshape(1)
	self.ctcloss = ctc.CTCLoss();

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
	acts = bottom[0].data
	print "DS2CtcLossLayer bot0 shape {}".format(acts.shape)
	targets = bottom[1].data
	input_percentages = bottom[2].data
	target_sizes = bottom[3].data
	sizes = input_percentages * acts.shape[0]
	self.ctcloss.ctc_loss(acts, targets, sizes.astype(np.int, copy=False), target_sizes)
	print "DS2CtcLossLayer self.costs type {} shape {}".format(type(self.ctcloss.costs), self.ctcloss.costs.shape)
	top[0].data[0] = self.ctcloss.costs.sum() / 20
	print "loss is {}".format(top[0].data[0])

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
	bottom[0].diff[...] = self.ctcloss.grad.astype(np.float32, copy=False)
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
