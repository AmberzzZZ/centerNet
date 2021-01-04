### formulation
    input: [h,w,3]
    output: x4 heatmap+offsets+size [h//s, w//s, c+2+2]
    backs: hourglass/resnet/DLA
    heads: independent 3x3 conv + 1x1 conv
    loss: 
        * det loss: grid focal loss, with penalty reduction on negatives
        * offset loss: l1, offh = h/s-h//s
        * size loss: l1, use raw pixels


    个人感觉，centerNet和anchor-based的formulation其实是一样的，
        - center的回归对标confidence的回归，区别在于高斯/[0,1]/[0,-1,1]
        - offset的回归对标bounding box中心点的回归，完全一样
        - size的回归变成了raw pixel，不再基于anchor
        - hourglass结构就是fpn，级联的hourglass可以对标bi-fpn
        - 多尺度变成了单一大resolution特征图，也可以用多尺度预测，需要加NMS


### hourglass
    reference: https://github.com/AmberzzZZ/backbones/tree/master/hourglass

    intermediate supervision:
        * centerNet论文里没提
        * cornerNet论文里提了：用了但是不作为下一级输入

    实验发现intermediate supervision非常重要！！！收敛快，而且heatmap更准


### one attemp
    centerNext: 因为发现hourglass中大量使用了residual block，考虑换成resNext block，提升计算效率
    参数量瞬间从33million下降到15million

    se-centerNext: one step further, add se-blocks in residuals








    
