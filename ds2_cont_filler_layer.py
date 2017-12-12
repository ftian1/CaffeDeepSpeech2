# --------------------------------------------------------
# Deep Speech 2 Caffe Implementation
# Written by Tian, Feng <feng.tian@intel.com>
# --------------------------------------------------------

"""The data layer used during training to train a DS2 network.

DS2ContFillerLayer implements a Caffe Python layer.
"""

import caffe
import numpy as np

class DS2ContFillerLayer(caffe.Layer):
    """DeepSpeech2 cont filler layer used for training."""

    def setup(self, bottom, top):
        """Setup the DS2ContFillerLayer."""
	top[0].reshape(bottom[0].shape[0], bottom[0].shape[1])

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
	top[0].data[...]  = 1
	top[0].data[:,0] = 0

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
