"""
3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation
Paper URL: https://arxiv.org/abs/1606.06650
Author: Amir Aghdam
"""

from torch import nn
from torchsummary import summary
import torch
import time
import torch.nn.functional as F
from config import FIRST_CHANNEL, BCE_WEIGHTS

import torch
import torch.nn as nn

class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.softmax(y_pred, dim=1)  # Softmax for multi-class
        y_true = nn.functional.one_hot(y_true, num_classes=y_pred.shape[1]).permute(0, 4, 1, 2, 3)   # One-hot encoding

        intersection = torch.sum(y_pred * y_true, dim=(2, 3, 4))  # (batch_size, NUM_CLASSES)
        dice = (2. * intersection + self.smooth) / (torch.sum(y_pred, dim=(2, 3, 4)) + torch.sum(y_true, dim=(2, 3, 4)) + self.smooth
        )
        return 1 - dice.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)  # 確保輸出值在 0-1 之間
        y_true = y_true.float()  # 確保標籤是浮點數

        intersection = torch.sum(y_pred * y_true)
        dice = (2. * intersection + self.smooth) / (torch.sum(y_pred) + torch.sum(y_true) + self.smooth)
        return 1 - dice  # 因為 Loss = 1 - Dice 係數


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=torch.Tensor(BCE_WEIGHTS))
        self.dice_loss = DiceLoss()
        self.alpha = alpha  # 調整交叉熵與 Dice Loss 的比重

    def forward(self, y_pred, y_true):
        return self.alpha * self.ce_loss(y_pred, y_true) + (1 - self.alpha) * self.dice_loss(y_pred, y_true)



class Conv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False, use_dropout=False, dropout_p=0.2) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels // 2, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels // 2)
        self.conv2 = nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout3d(p=dropout_p) if use_dropout else nn.Identity()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.dropout(res)
        res = self.relu(self.bn2(self.conv2(res)))
        res = self.dropout(res)
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer==False and num_classes==None) or (last_layer==True and num_classes!=None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2), stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels//2)
        self.conv1 = nn.Conv3d(in_channels=in_channels+res_channels, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels//2, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual!=None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out
        



class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[FIRST_CHANNEL, FIRST_CHANNEL*2, FIRST_CHANNEL*4], bottleneck_channel=FIRST_CHANNEL*8,use_dropout=True) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls,use_dropout=use_dropout)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls,use_dropout=use_dropout)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls,use_dropout=use_dropout)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck= True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls,)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls,)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes, last_layer=True)

    
    def forward(self, input):
        #Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        #Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)
        return out



if __name__ == '__main__':
    #Configurations according to the Xenopus kidney dataset
    model = UNet3D(in_channels=4, num_classes=1)
    start_time = time.time()
    # model.to("cuda")
    # print("--- %s seconds ---" % (time.time() - start_time))
    summary(model=model, input_size=(4, 240, 240, 160), batch_size=-1, device="cpu")
    print("--- %s seconds ---" % (time.time() - start_time))