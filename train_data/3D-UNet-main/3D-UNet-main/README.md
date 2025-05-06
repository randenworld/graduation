
# Pytorch implementation of 3D UNet

This implementation is based on the orginial 3D UNet paper and adapted to be used for MRI or CT image segmentation task   
> Link to the paper: [https://arxiv.org/pdf/1606.06650v1.pdf](https://arxiv.org/pdf/1606.06650v1.pdf)
> from [https://github.com/aghdamamir/3D-UNet.git]
## Model Architecture

The model architecture follows an encoder-decoder design which requires the input to be divisible by 16 due to its downsampling rate in the analysis path.

![3D Unet](https://github.com/AghdamAmir/3D-UNet/blob/main/3D-UNET.png)


## Configure the network

All the configurations and hyperparameters are set in the config.py file.
Please note that you need to change the path to the dataset directory in the config.py file before running the model.

## Training

After configure config.py, you can start to train by running

`python train.py`

We also employ tensorboard to visualize the training process.

`tensorboard --logdir=logs/`
