# Contribution

**PRs are always welcomed, please see TODOs and issues.**

## TODOs

### Documentation

* Improve documentation

### Features

* PyTorch-related:
    * PyTorch checkpoint and save model
    * Proxy `torch.nn.parameter.Parameter` for weight fields for optimizers
* Taichi related:
    * Wait for Taichi to have native PyTorch tensor view to optimize performance(i.e., no need to copy data back and
        forth)
    * Automatic Batching for `Tin` - waiting for upstream Taichi improvement
        * workaround for now: do static manual batching, that is to extend fields with one more dimension for batching
