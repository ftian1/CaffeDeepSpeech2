
# Caffe binding for WarpCTC

## Installation

Install [Caffe](https://github.com/intel/caffe.git).

`WARP_CTC_PATH` should be set to the location of a built WarpCTC
(i.e. `libwarpctc.so`).  This defaults to `../build`, so from within a
new warp-ctc clone you could build WarpCTC like this:

```bash
git clone https://github.com/baidu-research/warp-ctc.git
cd warp-ctc
mkdir build; cd build
cmake ..
make
```

Otherwise, set `WARP_CTC_PATH` to wherever you have `libwarpctc.so`
installed. 

```

Now install the bindings:
```
cd caffe_binding
python setup.py install
```

If you try the above and get a dlopen error on OSX with anaconda3 :
```
cd ../caffe_binding
python setup.py install
cd ../build
cp libwarpctc.dylib /Users/$WHOAMI/anaconda3/lib
```
This will resolve the library not loaded error. This can be easily modified to work with other python installs if needed.

Example to use the bindings below.

```python
    from warpctc_caffe import CTCLoss
    ctc_loss = CTCLoss()
    # expected shape of seqLength x batchSize x alphabet_size
    cost = ctc_loss(probs, labels, probs_sizes, label_sizes)
    cost.backward()
```
