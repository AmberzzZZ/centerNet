## centerNet

    official repo: https://github.com/xingyizhou/CenterNet/src

    input: [h,w,3]
    output: s4 heatmap+offsets+size [h//s, w//s, c+2+2]
    backs: hourglass/resnet/DLA
    heads: independent 3x3 conv + 1x1 conv
    loss: 
        * det loss: grid focal loss, with penalty reduction on negatives
        * offset loss: l1, offh = h/s-h//s, 中心点相对grid原点的offset
        * size loss: l1, 直接回归heatmap level的wh而不是offset(e.g. 如果input是512，那wh回归范围就是[0,128]), factor=0.1


    个人感觉，centerNet和anchor-based的formulation其实是一样的，
        - center的回归对标confidence的回归，区别在于高斯/[0,1]/[0,-1,1]
        - offset的回归对标bounding box中心点的回归，完全一样
        - size的回归变成了raw pixel，不再基于anchor
        - hourglass结构就是fpn，级联的hourglass可以对标bi-fpn
        - 多尺度变成了单一大resolution特征图，也可以用多尺度预测，需要加NMS

    MAP on COCO without TTA: 
    ResNet-101: 34.6 AP at 45 FPS
    DLA-34: 37.4% AP at 52 FPS 
    Hourglass-104: 45.1% AP at 1.4 FPS
    （加上test time multi-scale能涨5个点，说明single-scale是瓶颈）



### backbones

    * hourglass-104
    reference: https://github.com/AmberzzZZ/backbones/tree/master/hourglass
    stacked: 跟cornetNet里的结构一样，一个降采样的hourglass加上两个镜像对称的hourglass，参数量贼大
    intermediate supervision:
        - centerNet论文里没提
        - cornerNet论文里提了：用了但是不作为下一级输入
        - 实验发现intermediate supervision非常重要！！！收敛快，而且heatmap更准

    * resnet-101
    decoder design: 
        - channel数: [256,128,64]
        - 3x3 deformable conv + transpose conv(initialized as bilinear interpolation)

    * DLA-34
    decoder design: 
        - 与encoder镜像对称的结构
        - replace the original conv with 3x3 deformable conv at every upsampling layer
        - add more skip connections from each level
        - 与resnet back的主要不同是每个feature level融合了前一层的feature



    * one attemp
    centerNext: 因为发现hourglass中大量使用了residual block，考虑换成resNext block，提升计算效率
    参数量瞬间从33million下降到15million

    se-centerNext: one step further, add se-blocks in residuals

    compose augmentation: 
        label加权的不行（mixup），因为新定义的loss中positives必须prob为1，可以考虑cutmix和mosaic
        所以label smoothing也用不成



### head
    3x3 conv(256) + 1x1 conv


### test time decoder(from points to bnd boxes)

    for each heatmap channel:
    * take top 100 8-connected-neighbor-peaks: inplemented by 3x3 max pooling
    * 将peaks的xywh通道转化成box value
    * 论文声称无需任何NMS/其他后处理，但是top100本身就会带来冗余框，可能不影响precision，但是假阳就是很多啊？？



### training details

    * input size: 512x512x3
    * output: 128x128
    * aug: random flip & random scaling(0.6-1.3) & cropping & color jittering
    * Adam: batch size=128, lr=5e-4(drop by x10 at ep90 & ep120), 140 epochs, 







## centerNet2
    
    official repo: https://github.com/xingyizhou/CenterNet2

### main modifications from v1:
    
    * two-stage probablilistic model
        stage1是一个前/背景的RPN head，预测一个object heatmap，用了ResNet-FPN(ResNeXt-32x8d-101-DCN)，multi-scale预测，P3-P7
        stage2是一个CascadeRCNN head，只有object heatmap是前景的格子有预测，prob是centerness*stage1_prob

    * workflow params
        stage1的RPN因为更准了，所以只要top256个proposals
        stage2的proposal IoU thresh提高到[0.6, 0.7, 0.8]，用于从proposals里面区分前背景

    * loss
        cls loss
        - 论文里两阶段的训练和监督都是联动的
            - 拆分为前景loss和背景loss：因为只有stage1预测为前景才进行stage2的bnd预测
            - 论文里前景loss是个等式，背景loss是个不等式（通过优化两个bounds替代）
        - 但是我看代码里就是基于gt的focal loss各算各的
            - 包括stage2的：heatmap_focal_loss
            - 和stage1那个agn的：binary_heatmap_focal_loss

        reg loss: giou，这个没变

    * 骚操作堆叠
        groupNorm
        deformable conv
        giou
        bi-FPN
        Res2Net等ResNet变体
        FCOS的scale limit



### structure details
    
    1. agn: rpn branch，只负责生成一个heatmap，build over the box tower
    2. cls tower & box tower：两个独立的预测分支，4 convs+head，not shared，convs是conv with bias + groupNorm
    3. CascadeROIHeads: https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/cascade_rcnn.py
        - RPN提出的proposals大部分质量不高，因此没办法直接使用高iou阈值的detector
        - Cascade R-CNN使用cascade回归作为一种重采样的机制，逐stage提高proposal的IoU值
        - 每一个stage的detector都不会过拟合，而且不会出现train & test mismatch


### training details

    







    
