import numpy as np
cimport numpy as np
cimport cython

cdef extern int cpu_ctc(float *probs, int prob_size, float *grads, int *labels, int *label_sizes, int *sizes, int minibatch_size, float *costs);	

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
class CTCLoss(object):
	def __init__ (self):
		pass
	def ctc_loss(self, acts, labels, act_lens, label_lens):
		#print "acts : {}".format(acts.shape)
		#print acts
		#print "labels : {}".format(labels.shape)
		#print labels
		#print "act_lens : {}".format(act_lens.shape)
		#print act_lens
		#print "label_lens : {}".format(label_lens.shape)
		#print label_lens
		assert len(labels.shape) == 1 # labels must be 1 dimensional
		minibatch_size = acts.shape[1]
		acts_size = acts.shape[2]

		cdef np.ndarray[np.float32_t, ndim=3, mode="c"] acts_c
		cdef np.ndarray[np.int32_t, ndim=1, mode="c"] labels_c
		cdef np.ndarray[np.int32_t, ndim=1, mode="c"] act_lens_c
		cdef np.ndarray[np.int32_t, ndim=1, mode="c"] label_lens_c

		acts_c = np.ascontiguousarray(acts, dtype=np.float32)
		cdef float *acts_c_ptr = &acts_c[0,0,0]
		labels_c = np.ascontiguousarray(labels, dtype=np.int32)
		cdef int *labels_c_ptr = <int *>&labels_c[0]
		act_lens_c = np.ascontiguousarray(act_lens, dtype=np.int32)
		cdef int *act_lens_c_ptr = <int *>&act_lens_c[0]
		label_lens_c = np.ascontiguousarray(label_lens, dtype=np.int32)
		cdef int *label_lens_c_ptr = <int *>&label_lens_c[0]

		cdef np.ndarray[np.float32_t, ndim=3, mode="c"] grads = np.zeros(acts.shape, dtype=np.float32)
		cdef float *grads_ptr = &grads[0,0,0]
		cdef np.ndarray[np.float32_t, ndim=1, mode="c"] costs = np.zeros(minibatch_size, dtype=np.float32)
		cdef float *costs_ptr = &costs[0]
		self.grads = grads
		self.costs = costs
		#print "shape is {}".format(labels_c.shape)
		#print "val is {} {}".format(labels_c[0], labels_c[1])
		#print labels_c_ptr[0]
		#print labels_c_ptr[1]

		cpu_ctc(acts_c_ptr,
                        acts_size,
                        grads_ptr,
                        labels_c_ptr,
                        label_lens_c_ptr,
                        act_lens_c_ptr,
                        minibatch_size,
                        costs_ptr)
		return self.costs.sum()
