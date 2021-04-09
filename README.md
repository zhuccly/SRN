# MDM

A Tensorflow implementation of the structural relation network.


# Installation Instructions


## Menpo 0.8.1
## Menpodetect 0.5.0
## Menpo fit 0.5.0
we use the Menpo project in various ways throughout the implementation.

Please look at the installation instructions at:

    http://www.menpo.org/installation/

## TensorFlow 1.10.1

# Pretrained models

The pre-training model is coming soon.

# Training a model
Currently the TensorFlow implementation does not contain tracking model we did in the submitted paper, but this will be updated shortly.

```
    # Activate the conda environment.
    source activate environment-name
    
    # Start training
    python mdm_train.py --datasets='databases/lfpw/trainset/*.png:databases/afw/*.jpg:databases/helen/trainset/*.jpg'
    
    # Track the train process and evaluate the current checkpoint against the validation set
    python mdm_eval.py --dataset_path="./databases/ibug/*.jpg" --num_examples=135 --eval_dir=ckpt/eval_ibug  --device='/cpu:0' --checkpoint_dir=$PWD/ckpt/train
    
    python mdm_eval.py --dataset_path="./databases/lfpw/testset/*.png" --num_examples=300 --eval_dir=ckpt/eval_lfpw  --device='/cpu:0' --checkpoint_dir=$PWD/ckpt/train
    
    python mdm_eval.py --dataset_path="./databases/helen/testset/*.jpg" --num_examples=330 --eval_dir=ckpt/eval_helen  --device='/cpu:0' --checkpoint_dir=$PWD/ckpt/train
    
    # Run tensorboard to visualise the results
    tensorboard --logdir==$PWD/ckpt
```
The realization of some functions refers to the MDM project (https://github.com/trigeorgis/mdm).

