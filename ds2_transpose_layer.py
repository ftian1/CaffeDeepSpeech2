# --------------------------------------------------------
# Deep Speech 2 Caffe Implementation
# Written by Tian, Feng <feng.tian@intel.com>
# --------------------------------------------------------

"""The data layer used during training to train a DS2 network.

DS2TransposeLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np

class DS2TransposeLayer(caffe.Layer):
    """DeepSpeech2 transpose layer used for training."""

    def setup(self, bottom, top):
        """Setup the DS2TransposeLayer."""
	n = bottom[0].shape[0]
	c = bottom[0].shape[1]
	h = bottom[0].shape[2]
	w = bottom[0].shape[3]
	top[0].reshape(w, n, c*h)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
	n = bottom[0].shape[0]
	c = bottom[0].shape[1]
	h = bottom[0].shape[2]
	w = bottom[0].shape[3]
	top[0].reshape(w, n, c*h)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def transpose(self, bottom, top):
	n = bottom[0].shape[0]
	c = bottom[0].shape[1]
	h = bottom[0].shape[2]
	w = bottom[0].shape[3]
	top[0].reshape(w, n, c*h)
        
