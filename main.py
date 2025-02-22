import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]

import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
)
from tqdm import tqdm
from torchvision import utils
from hparam import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir


source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

output_dir_test = hp.output_dir_test



def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file, help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')  
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    return parser



def train():
    # 创建参数解析器，并设置描述信息
    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    # 调用自定义函数添加训练相关参数
    parser = parse_training_args(parser)
    # 初步解析已知参数（允许存在未定义的参数），返回已解析参数和未解析参数列表
    args, _ = parser.parse_known_args()
    # 再次完整解析所有参数（会覆盖之前的args变量）
    args = parser.parse_args()

    # 设置PyTorch的cuDNN后端配置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    # 延迟导入数据模块（可能需要先解析某些路径参数）
    from data_function import MedData_train
    # 创建输出目录（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        #from models.two_d.unet import Unet
        #model = Unet(in_channels=hp.in_class, classes=hp.out_class+1)

        from models.two_d.miniseg import MiniSeg
        model = MiniSeg(in_input=hp.in_class, classes=hp.out_class+1)

        #from models.two_d.fcn import FCN32s as fcn
        #model = fcn(in_class =hp.in_class,n_class=hp.out_class+1)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class+1)

        #from models.two_d.deeplab import DeepLabV3
        #model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class+1)

        #from models.two_d.unetpp import ResNet34UnetPlus
        #model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class+1)

        #from models.two_d.pspnet import PSPNet
        #model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class+1)

    elif hp.mode == '3d':
        # 从three_d子模块导入标准的3D U-Net实现
        #from models.three_d.unet3d import UNet3D
        #model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class+1, init_features=32)

        # 当前使用的带残差连接的3D U-Net变体
        from models.three_d.residual_unet3d import UNet
        # 实例化残差U-Net模型
        model = UNet(in_channels=hp.in_class, n_classes=hp.out_class+1, base_n_filter=2)

        #from models.three_d.fcn3d import FCN_Net
        #model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class+1)

        #from models.three_d.highresnet import HighRes3DNet
        #model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class+1)

        #from models.three_d.densenet3d import SkipDenseNet3D
        #model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class+1)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class+1)

        #from models.three_d.vnet3d import VNet
        #model = VNet(in_channels=hp.in_class, classes=hp.out_class+1)

        #from models.three_d.unetr import UNETR
        #model = UNETR(img_shape=(hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class+1)

    # 将模型包装为DataParallel，实现多GPU并行计算
    # device_ids参数指定使用的GPU设备列表
    model = torch.nn.DataParallel(model, device_ids=devicess)
    # 定义优化器，使用Adam算法
    # model.parameters()获取模型所有可训练参数
    # lr=args.init_lr设置初始学习率，从命令行参数中获取
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

    # （注释掉的学习率调度器1：基于验证损失的动态调整）
    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    # 当前使用的学习率调度器：StepLR
    # 按固定步长调整学习率
    # step_size=hp.scheduer_step_size：从超参数中获取调整步长（单位：epoch）
    # gamma=hp.scheduer_gamma：从超参数中获取学习率衰减系数
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    # 将模型移动到GPU上运行
    model.cuda()

    # 导入损失函数模块
    # Binary_Loss: 自定义的二元分类损失函数
    # DiceLoss: 自定义的Dice损失函数，常用于医学图像分割
    from loss_function import Binary_Loss,DiceLoss
    # 导入PyTorch内置的交叉熵损失函数
    from torch.nn.modules.loss import CrossEntropyLoss
    # 初始化Dice损失函数
    # DiceLoss(2): 创建一个Dice损失函数实例，参数2可能表示类别数
    # .cuda(): 将损失函数移到GPU上
    criterion_dice = DiceLoss(2).cuda()
    # 初始化交叉熵损失函数
    # CrossEntropyLoss(): 创建交叉熵损失函数实例
    # .cuda(): 将损失函数移到GPU上
    criterion_ce = CrossEntropyLoss().cuda()

    # 创建TensorBoard日志记录器
    # SummaryWriter: PyTorch提供的可视化工具接口
    # args.output_dir: 日志保存路径，从命令行参数中获取
    writer = SummaryWriter(args.output_dir)

    # 创建训练数据集实例
    # MedData_train: 自定义的医学图像数据集类
    # source_train_dir: 训练图像数据目录
    # label_train_dir: 对应的标签数据目录
    train_dataset = MedData_train(source_train_dir,label_train_dir)
    # 创建数据加载器
    # train_dataset.queue_dataset: 获取数据集的可迭代对象
    # batch_size=args.batch: 从命令行参数中获取批量大小
    # shuffle=True: 每个epoch打乱数据顺序
    # pin_memory=True: 将数据加载到CUDA固定内存中，加速GPU传输
    # drop_last=True: 丢弃最后一个不完整的batch
    train_loader = DataLoader(train_dataset.queue_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    # 将模型设置为训练模式
    # 启用dropout和batch normalization等训练特定行为
    model.train()

    # 计算剩余训练轮数
    # args.epochs: 总训练轮数
    # elapsed_epochs: 已完成的训练轮数
    epochs = args.epochs - elapsed_epochs
    # 计算当前迭代次数
    # len(train_loader): 每个epoch的迭代次数（batch数量）
    # elapsed_epochs * len(train_loader): 已完成的迭代次数
    iteration = elapsed_epochs * len(train_loader)

    # 开始训练循环，遍历每个epoch
    for epoch in range(1, epochs + 1):
        # 打印当前epoch
        print("epoch:"+str(epoch))
        # 更新全局epoch计数（考虑从断点恢复训练）
        epoch += elapsed_epochs

        # 初始化当前epoch的迭代计数器
        num_iters = 0

        # 遍历数据加载器中的每个batch
        for i, batch in enumerate(train_loader):

            # 调试模式：仅运行一个batch后退出
            if hp.debug:
                if i >=1:
                    break
            # 打印当前batch和epoch信息
            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")
            # 清除优化器的梯度
            optimizer.zero_grad()

            # 处理二分类任务（输入和输出均为单通道）
            if (hp.in_class == 1) and (hp.out_class == 1) :
                # 从batch中提取输入数据和标签
                x = batch['source']['data']
                y = batch['label']['data']


                #y[y!=0] = 1
                # 创建背景标签（y_back），标记非目标区域
                y_back = torch.zeros_like(y)
                # y_back[(y==0) ^ (y_L_TL==0) ^ (y_R_TL==0)]=1
                # 标记背景区域（y == 0 的区域）
                y_back[(y==0)]=1

                # 将输入数据转换为FloatTensor并移动到GPU
                x = x.type(torch.FloatTensor).cuda()
                # 将背景标签和目标标签拼接在一起
                y = torch.cat((y_back, y),1)
                # 将标签数据转换为FloatTensor并移动到GPU
                y = y.type(torch.FloatTensor).cuda()
                
            else:
                x = batch['source']['data']
                y_atery = batch['atery']['data']
                y_lung = batch['lung']['data']
                y_trachea = batch['trachea']['data']
                y_vein = batch['atery']['data']


                y_back = torch.zeros_like(y_atery)
                y_back[(y_atery==0) ^ (y_lung==0) ^ (y_trachea==0) ^ (y_vein==0)]=1


                x = x.type(torch.FloatTensor).cuda()

                y = torch.cat((y_back,y_atery,y_lung,y_trachea,y_vein),1) 
                y = y.type(torch.FloatTensor).cuda()


            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)

            # 将标签 y 中所有非零值设为 1（二值化处理）
            y[y!=0] = 1
                
                #print(y.max())

            # 前向传播：将输入数据 x 传入模型，得到输出 outputs
            outputs = model(x)


            # for metrics
            # 计算模型预测的类别标签（argmax 获取每个像素点的预测类别）
            labels = outputs.argmax(dim=1)
            # 将预测标签转换为 one-hot 编码，用于计算指标
            # num_classes=hp.out_class+1：类别数（包括背景）
            # permute(0,4,1,2,3)：调整维度顺序以匹配输出格式
            model_output_one_hot = torch.nn.functional.one_hot(labels, num_classes=hp.out_class+1).permute(0,4,1,2,3)

            # 计算损失函数
            # criterion_ce：交叉熵损失
            # criterion_dice：Dice 损失
            # y.argmax(dim=1)：将 one-hot 标签转换为类别索引
            loss = criterion_ce(outputs, y.argmax(dim=1)) + criterion_dice(outputs, y.argmax(dim=1))
            # loss = criterion_ce(outputs, y) + criterion_dice(outputs, y.argmax(dim=1).unsqueeze(1))
            # loss = criterion_ce(outputs, y.argmax(dim=1)) + criterion_dice(outputs, y.argmax(dim=1).unsqueeze(1))



            # logits = torch.sigmoid(outputs)
            # labels = logits.clone()
            # labels[labels>0.5] = 1
            # labels[labels<=0.5] = 0

            # 更新迭代计数器
            num_iters += 1
            # 反向传播：计算梯度
            loss.backward()

            # 更新模型参数
            optimizer.step()
            # 更新全局迭代计数器
            iteration += 1

            # 将真实标签 y 转换为类别索引（argmax）
            y_argmax = y.argmax(dim=1)
            # 将类别索引转换为 one-hot 编码
            # num_classes=hp.out_class+1：类别数（包括背景）
            # permute(0,4,1,2,3)：调整维度顺序以匹配模型输出格式
            y_one_hot = torch.nn.functional.one_hot(y_argmax, num_classes=hp.out_class+1).permute(0,4,1,2,3)

            # 计算指标：假阳性率、假阴性率、Dice 系数
            # y_one_hot[:,1:,:,:]：忽略背景类别（第0维）
            # model_output_one_hot[:,1:,:,:]：忽略背景类别（第0维）
            # .cpu()：将数据移动到 CPU 以进行计算
            false_positive_rate,false_negtive_rate,dice = metric(y_one_hot[:,1:,:,:].cpu(),model_output_one_hot[:,1:,:,:].cpu())
    



            # false_positive_rate,false_negtive_rate,dice = metric(y.cpu(),labels.cpu())
            ## log
            # （注释掉的另一种指标计算方式）
            # false_positive_rate,false_negtive_rate,dice = metric(y.cpu(),labels.cpu())

            # 记录训练日志
            # 将损失值、假阳性率、假阴性率、Dice 系数写入 TensorBoard
            writer.add_scalar('Training/Loss', loss.item(),iteration)
            writer.add_scalar('Training/false_positive_rate', false_positive_rate,iteration)
            writer.add_scalar('Training/false_negtive_rate', false_negtive_rate,iteration)
            writer.add_scalar('Training/dice', dice,iteration)

            # 打印当前损失值
            print("loss:"+str(loss.item()))
            # 打印当前学习率
            # scheduler._last_lr[0]：获取当前学习率
            print('lr:'+str(scheduler._last_lr[0]))

        # 更新学习率调度器
        # 根据当前 epoch 或指标值调整学习率
        scheduler.step()


        # Store latest checkpoint in each epoch
        # 保存当前 epoch 的最新检查点
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            # 检查点保存路径
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )




        # Save checkpoint
        # 定期保存检查点
        # 每隔 args.epochs_per_checkpoint 个 epoch 保存一次
        if epoch % args.epochs_per_checkpoint == 0:
            # 保存检查点
            torch.save(
                {
                    
                    "model": model.state_dict(),# 保存模型参数
                    "optim": optimizer.state_dict(),# 保存优化器状态
                    "epoch": epoch,# 保存当前 epoch 数
                },
                # 检查点保存路径，文件名包含 epoch 数（格式化为4位数）
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )

            # 在不计算梯度的上下文中执行以下操作
            with torch.no_grad():
                # 如果模式是 2D，增加一个维度以适应 3D 数据的格式
                if hp.mode == '2d':
                    x = x.unsqueeze(4) # 在最后一个维度上增加一维
                    y = y.unsqueeze(4)
                    outputs = outputs.unsqueeze(4)

                # 将数据从 GPU 移动到 CPU，并转换为 NumPy 数组
                x = x[0].cpu().detach().numpy() # 取 batch 中的第一个样本
                y = y[0].cpu().detach().numpy()
                outputs = outputs[0].cpu().detach().numpy()
                model_output_one_hot = model_output_one_hot[0].float().cpu().detach().numpy()
                affine = batch['source']['affine'][0].numpy() # 获取仿射变换矩阵

                # 如果是二分类任务
                if (hp.in_class == 1) and (hp.out_class == 1) :
                    # 增加一个维度以适应 torchio 的输入格式
                    y = np.expand_dims(y, axis=1)
                    outputs = np.expand_dims(outputs, axis=1)
                    model_output_one_hot = np.expand_dims(model_output_one_hot, axis=1)

                    # 保存输入图像
                    source_image = torchio.ScalarImage(tensor=x, affine=affine)
                    source_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-source"+hp.save_arch))
                    # source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

                    # 保存真实标签图像
                    label_image = torchio.ScalarImage(tensor=y[1], affine=affine)
                    label_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt"+hp.save_arch))

                    # 保存预测结果图像
                    output_image = torchio.ScalarImage(tensor=model_output_one_hot[1], affine=affine)
                    output_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict"+hp.save_arch))
                # 如果是多分类任务
                else:
                    # 增加一个维度以适应 torchio 的输入格式
                    y = np.expand_dims(y, axis=1)
                    outputs = np.expand_dims(outputs, axis=1)

                    # 保存输入图像
                    source_image = torchio.ScalarImage(tensor=x, affine=affine)
                    source_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-source"+hp.save_arch))

                    # 分别保存每个类别的真实标签和预测结果
                    # 动脉
                    label_image_artery = torchio.ScalarImage(tensor=y[0], affine=affine)
                    label_image_artery.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_artery"+hp.save_arch))

                    output_image_artery = torchio.ScalarImage(tensor=outputs[0], affine=affine)
                    output_image_artery.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_artery"+hp.save_arch))

                    # 肺
                    label_image_lung = torchio.ScalarImage(tensor=y[1], affine=affine)
                    label_image_lung.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_lung"+hp.save_arch))

                    output_image_lung = torchio.ScalarImage(tensor=outputs[1], affine=affine)
                    output_image_lung.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_lung"+hp.save_arch))

                    # 气管
                    label_image_trachea = torchio.ScalarImage(tensor=y[2], affine=affine)
                    label_image_trachea.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_trachea"+hp.save_arch))

                    output_image_trachea = torchio.ScalarImage(tensor=outputs[2], affine=affine)
                    output_image_trachea.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_trachea"+hp.save_arch))

                    # 静脉
                    label_image_vein = torchio.ScalarImage(tensor=y[3], affine=affine)
                    label_image_vein.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt_vein"+hp.save_arch))

                    output_image_vein = torchio.ScalarImage(tensor=outputs[3], affine=affine)
                    output_image_vein.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict_vein"+hp.save_arch))

    # 关闭 TensorBoard 的 SummaryWriter
    writer.close()


def test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from data_function import MedData_test

    os.makedirs(output_dir_test, exist_ok=True)

    if hp.mode == '2d':
        #from models.two_d.unet import Unet
        #model = Unet(in_channels=hp.in_class, classes=hp.out_class+1)

        from models.two_d.miniseg import MiniSeg
        model = MiniSeg(in_input=hp.in_class, classes=hp.out_class+1)

        #from models.two_d.fcn import FCN32s as fcn
        #model = fcn(in_class =hp.in_class,n_class=hp.out_class+1)

        # from models.two_d.segnet import SegNet
        # model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class+1)

        #from models.two_d.deeplab import DeepLabV3
        #model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class+1)

        #from models.two_d.unetpp import ResNet34UnetPlus
        #model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class+1)

        #from models.two_d.pspnet import PSPNet
        #model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class+1)

    elif hp.mode == '3d':
        #from models.three_d.unet3d import UNet3D
        #model = UNet3D(in_channels=hp.in_class, out_channels=hp.out_class+1, init_features=32)

        from models.three_d.residual_unet3d import UNet
        model = UNet(in_channels=hp.in_class, n_classes=hp.out_class+1, base_n_filter=2)

        #from models.three_d.fcn3d import FCN_Net
        #model = FCN_Net(in_channels =hp.in_class,n_class =hp.out_class+1)

        #from models.three_d.highresnet import HighRes3DNet
        #model = HighRes3DNet(in_channels=hp.in_class,out_channels=hp.out_class+1)

        #from models.three_d.densenet3d import SkipDenseNet3D
        #model = SkipDenseNet3D(in_channels=hp.in_class, classes=hp.out_class+1)

        # from models.three_d.densevoxelnet3d import DenseVoxelNet
        # model = DenseVoxelNet(in_channels=hp.in_class, classes=hp.out_class+1)

        #from models.three_d.vnet3d import VNet
        #model = VNet(in_channels=hp.in_class, classes=hp.out_class+1)

        #from models.three_d.unetr import UNETR
        #model = UNETR(img_shape=(hp.crop_or_pad_size), input_dim=hp.in_class, output_dim=hp.out_class+1)


    model = torch.nn.DataParallel(model, device_ids=devicess)


    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])


    model.cuda()



    test_dataset = MedData_test(source_test_dir,label_test_dir)
    znorm = ZNormalization()

    if hp.mode == '3d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size
    elif hp.mode == '2d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size


    for i,subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
                subj,
                patch_size,
                patch_overlap,
            )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=args.batch)
        # aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
        model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):


                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                if hp.mode == '2d':
                    input_tensor = input_tensor.squeeze(4)
                outputs = model(input_tensor)

                if hp.mode == '2d':
                    outputs = outputs.unsqueeze(4)

                labels = outputs.argmax(dim=1)
                # model_output_one_hot = torch.nn.functional.one_hot(labels, num_classes=hp.out_class+1).permute(0,4,1,2,3)
                # logits = torch.sigmoid(outputs)

                # labels = logits.clone()
                # labels[labels>0.5] = 1
                # labels[labels<=0.5] = 0

                # aggregator.add_batch(model_output_one_hot, locations)
                aggregator_1.add_batch(labels.unsqueeze(1), locations)
        # output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()




        affine = subj['source']['affine']
        if (hp.in_class == 1) and (hp.out_class == 1) :
            # label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
            # label_image.save(os.path.join(output_dir_test,f"{i:04d}-result_float"+hp.save_arch))

            # f"{str(i):04d}-result_float.mhd"

            output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
            output_image.save(os.path.join(output_dir_test,f"{i:04d}-result_int"+hp.save_arch))
        else:
            output_tensor = output_tensor.unsqueeze(1)
            output_tensor_1= output_tensor_1.unsqueeze(1)

            output_image_artery_float = torchio.ScalarImage(tensor=output_tensor[0].numpy(), affine=affine)
            output_image_artery_float.save(os.path.join(output_dir_test,f"{i:04d}-result_float_artery"+hp.save_arch))
            # f"{str(i):04d}-result_float_artery.mhd"

            output_image_artery_int = torchio.ScalarImage(tensor=output_tensor_1[0].numpy(), affine=affine)
            output_image_artery_int.save(os.path.join(output_dir_test,f"{i:04d}-result_int_artery"+hp.save_arch))

            output_image_lung_float = torchio.ScalarImage(tensor=output_tensor[1].numpy(), affine=affine)
            output_image_lung_float.save(os.path.join(output_dir_test,f"{i:04d}-result_float_lung"+hp.save_arch))
            

            output_image_lung_int = torchio.ScalarImage(tensor=output_tensor_1[1].numpy(), affine=affine)
            output_image_lung_int.save(os.path.join(output_dir_test,f"{i:04d}-result_int_lung"+hp.save_arch))

            output_image_trachea_float = torchio.ScalarImage(tensor=output_tensor[2].numpy(), affine=affine)
            output_image_trachea_float.save(os.path.join(output_dir_test,f"{i:04d}-result_float_trachea"+hp.save_arch))

            output_image_trachea_int = torchio.ScalarImage(tensor=output_tensor_1[2].numpy(), affine=affine)
            output_image_trachea_int.save(os.path.join(output_dir_test,f"{i:04d}-result_int_trachea"+hp.save_arch))

            output_image_vein_float = torchio.ScalarImage(tensor=output_tensor[3].numpy(), affine=affine)
            output_image_vein_float.save(os.path.join(output_dir_test,f"{i:04d}-result_float_vein"+hp.save_arch))

            output_image_vein_int = torchio.ScalarImage(tensor=output_tensor_1[3].numpy(), affine=affine)
            output_image_vein_int.save(os.path.join(output_dir_test,f"{i:04d}-result_int_vein"+hp.save_arch))           


   

if __name__ == '__main__':
    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        test()
