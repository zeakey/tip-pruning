import torch, torchvision
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os, sys, argparse, time, shutil
from os.path import join, split, isdir, isfile, dirname, abspath
from vltools import Logger, run_path
from vltools import image as vlimage
from vltools.pytorch import save_checkpoint, AverageMeter, accuracy
from vltools.pytorch.datasets import ilsvrc2012
import vltools.pytorch as vlpytorch
from torch.optim.lr_scheduler import MultiStepLR
from utils.get_threshold import get_threshold

from models import preact_resnet_cifar
from tensorboardX import SummaryWriter

from thop import profile

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--model', metavar='STR', default=None, help='model')
parser.add_argument('--data', metavar='DIR', default=None, help='path to dataset')
parser.add_argument('--dataset', default="cifar100", help='dataset')
parser.add_argument('--bs', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--stepsize', '--step-size', default=None, type=int,
                    metavar='SS', help='decrease learning rate every stepsize epochs')
parser.add_argument('--gamma', default=0.1, type=float,
                    metavar='GM', help='decrease learning rate by gamma')
parser.add_argument('--milestones', default="80,120", type=str)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--resume', default="", type=str, metavar='PATH',
                    help='path to latest checkpoint')
parser.add_argument('--tmp', help='tmp folder', default="tmp/prune")
parser.add_argument('--randseed', type=int, help='random seed', default=None)
#
parser.add_argument('--rand-init-bn', action="store_true")
parser.add_argument('--rescalebn', action="store_true")
parser.add_argument('--fix-lr', action="store_true")
parser.add_argument('--no-retrain', action="store_true")
parser.add_argument('--sparsity', type=float, default=1e-5, help='sparsity regularization')
parser.add_argument('---delta-lambda', type=float, default=1e-5, help='delta lambda')
parser.add_argument('--sparse-thres', type=float, default=2e-2, help='sparse threshold')
parser.add_argument('--retrain', action="store_true")
parser.add_argument('--retrain-epochs', type=int, default=160, help="retrain epochs")
parser.add_argument('--depth', type=int, default=164, help='model depth')
parser.add_argument('--percent', type=float, default=0.5, help='pruning percent')
args = parser.parse_args()

milestones = [int(i) for i in args.milestones.split(',')]

args.sparsity = 0
args.fix_lr = True

if args.randseed == None:
    args.randseed = np.random.randint(1000)

args.tmp = args.tmp.strip("/")
args.tmp = "tmp"
args.tmp = run_path(args.tmp)

# Random seed
# According to https://pytorch.org/docs/master/notes/randomness.html
np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.cuda.manual_seed_all(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

THIS_DIR = abspath(dirname(__file__))
os.makedirs(args.tmp, exist_ok=True)

# loss function
criterion = torch.nn.CrossEntropyLoss()

tfboard_writer = writer = SummaryWriter(log_dir=args.tmp)
logger = Logger(join(args.tmp, "log.txt"))

from models.preact_resnet_cifar import Bottleneck

@torch.no_grad()
def get_factors(model):
    factors = {}
    for name, m in model.named_modules():
        if isinstance(m, Bottleneck):
            
            name1 = name+".bn1"
            factor = m.bn1.weight.data.abs()
            next_weight = m.conv1.weight.abs()
            next_weight = next_weight.mean(dim=(0,2,3))
            factor = factor * next_weight
            factors[name1] = factor

            name1 = name+".bn2"
            factor = m.bn2.weight.data.abs()
            next_weight = m.conv2.weight.abs()
            next_weight = next_weight.mean(dim=(0,2,3))
            factor = factor * next_weight
            factors[name1] = factor

            name1 = name+".bn3"
            factor = m.bn3.weight.data.abs()
            next_weight = m.conv3.weight.abs()
            next_weight = next_weight.mean(dim=(0,2,3))
            factor = factor * next_weight
            factors[name1] = factor


    return factors

def get_sparsity(factors, thres):
    total0 = 0
    total = 0
    for v in factors.values():
        total0 += (v <= v.max()*thres).sum()
        total += v.numel()
    return (total0.float() / total).item()

def get_sparsity_from_model(model, thres):
    return get_sparsity(get_factors(model), thres)

@torch.no_grad()
def mask_prune(model, factors, thres):
    prune_mask = {}
    total_filters = 0
    pruned_filters = 0
    for name, m in model.named_modules():

        if isinstance(m, Bottleneck):
            for name1, m1 in m.named_modules():
                if isinstance(m1, nn.BatchNorm2d):
                    name2 = name+"."+name1
                    factor = factors[name2]
                    
                    mask = factor >= factor.max() * thres

                    # in case of too many channels being pruned
                    min_channels = 8
                    if mask.sum() <= min_channels:
                        _, idx = factor.sort(descending=True)
                        idx = idx[0:min_channels]
                        mask[idx] = 1

                        num_pruned = mask.numel()-mask.sum()

                        logger.info("Warning: layer %s, %d filters pruned."
                                "Manually preserve %d filters."%(name1, num_pruned, min_channels))

                    m1.weight.data[mask.bitwise_not()] = 0
                    m1.bias.data[mask.bitwise_not()] = 0
                    m1.running_mean[mask.bitwise_not()] = 0
                    m1.running_var[mask.bitwise_not()] = 1

                    prune_mask[name2] = mask
                    total_filters += float(mask.numel())
                    pruned_filters += float((mask.bitwise_not()).sum().item())

                    logger.info("Layer %s remaining filters %d (original %d)"%\
                                (name2, mask.sum(), mask.numel()))
    logger.info("--------------------> %d of %d filters pruned, pruning rate%f<--------------------"%\
               (pruned_filters,total_filters, pruned_filters/total_filters))

    return prune_mask

@torch.no_grad()
def real_prune(model, prune_mask):
    for name, m in model.named_modules():
        if isinstance(m, Bottleneck):
            mask1 = prune_mask[name+".bn1"]
            mask2 = prune_mask[name+".bn2"]
            mask3 = prune_mask[name+".bn3"]
            m.select.indices.zero_()
            m.select.indices[mask1] = 1
            # # BN1
            new_bn = nn.BatchNorm2d(int(mask1.numel()))
            new_bn.weight.data.copy_(m.bn1.weight.data)
            new_bn.bias.data.copy_(m.bn1.bias.data)
            new_bn.running_mean.copy_(m.bn1.running_mean)
            new_bn.running_var.copy_(m.bn1.running_var)
            del m.bn1
            m.bn1 = new_bn
            # BN2
            new_bn = nn.BatchNorm2d(int(mask2.sum()))
            new_bn.weight.data.copy_(m.bn2.weight.data[mask2])
            new_bn.bias.data.copy_(m.bn2.bias.data[mask2])
            new_bn.running_mean.copy_(m.bn2.running_mean[mask2])
            new_bn.running_var.copy_(m.bn2.running_var[mask2])
            del m.bn2
            m.bn2 = new_bn
            # BN3
            new_bn = nn.BatchNorm2d(int(mask3.sum()))
            new_bn.weight.data.copy_(m.bn3.weight.data[mask3])
            new_bn.bias.data.copy_(m.bn3.bias.data[mask3])
            new_bn.running_mean.copy_(m.bn3.running_mean[mask3])
            new_bn.running_var.copy_(m.bn3.running_var[mask3])
            del m.bn3
            m.bn3 = new_bn

            # Conv1
            in_channels = int(mask1.sum())
            out_channels = int(mask2.sum())
            new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=m.conv1.kernel_size,
                                 padding=m.conv1.padding, stride=m.conv1.stride, bias=False)
            new_conv.weight.data.copy_(m.conv1.weight.data[mask2][:, mask1])
            del m.conv1
            m.conv1 = new_conv

            # Conv2
            in_channels = int(mask2.sum())
            out_channels = int(mask3.sum())
            new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=m.conv2.kernel_size,
                                 padding=m.conv2.padding, stride=m.conv2.stride, bias=False)
            new_conv.weight.data.copy_(m.conv2.weight.data[mask3,][:, mask2,])
            del m.conv2
            m.conv2 = new_conv

            # Conv3
            in_channels = int(mask3.sum())
            out_channels = m.conv3.out_channels
            new_conv = nn.Conv2d(in_channels, out_channels, kernel_size=m.conv3.kernel_size,
                                 padding=m.conv3.padding, stride=m.conv3.stride, bias=False)
            new_conv.weight.data.copy_(m.conv3.weight.data[:, mask3])
            del m.conv3
            m.conv3 = new_conv

    return model

def main():

    logger.info(args)

    if args.dataset == "cifar10":
        train_loader, val_loader = vlpytorch.datasets.cifar10(abspath("datasets/"), bs=args.bs)
        num_classes = 10
    elif args.dataset == "cifar100":
        train_loader, val_loader = vlpytorch.datasets.cifar100(abspath("datasets/"), bs=args.bs)
        num_classes = 100
    elif args.dataset == "svhn":
        train_loader, val_loader = vlpytorch.datasets.svhn(abspath("datasets/"), bs=args.bs)
        num_classes = 10
    else:
        raise ValueError("Invalid dataset %s" % args.dataset)

    # model and optimizer
    model_name = "preact_resnet_cifar.resnet(depth=%d, num_classes=%d)" % (args.depth, num_classes)
    model = eval(model_name).cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), ), verbose=False)
    tfboard_writer.add_scalar("train/FLOPs", flops, -1)
    tfboard_writer.add_scalar("train/Params", params, -1)
    model = nn.DataParallel(model)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    logger.info("Model details:")
    logger.info(model)
    logger.info("Optimizer details:")
    logger.info(optimizer)

    # records
    best_acc1 = 0

    if args.rand_init_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, 1)

    # save initial weights
    save_checkpoint({
            'epoch': -1,
            'state_dict': model.state_dict(),
            'best_acc1': -1,
            }, False, path=args.tmp, filename="initial-weights.pth")

    # optionally resume from a checkpoint
    if args.resume:
        if isfile(args.resume):
            shutil.copy(args.resume, args.tmp)
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    last_sparsity = get_sparsity(get_factors(model), thres=args.sparse_thres)
    for k, v in get_factors(model).items():
            tfboard_writer.add_histogram("train/%s"%k, v.cpu().numpy(), -1)
    for epoch in range(args.start_epoch, args.epochs):

        # train and evaluate
        loss = train(train_loader, model, optimizer, epoch, l1lambda=args.sparsity)
        acc1, acc5 = validate(val_loader, model, epoch)
        if not args.fix_lr:
            scheduler.step()

        # calculate FLOPs and params
        m = eval(model_name)
        factors = get_factors(model.module)
        
        thres = args.sparse_thres
        prune_mask = mask_prune(m, factors, thres)
        m = real_prune(m, prune_mask)
        flops, params = profile(m, inputs=(torch.randn(1, 3, 32, 32), ), verbose=False)
        del m
        logger.info("%d FLOPs, %d params"%(flops, params))
        tfboard_writer.add_scalar("train/FLOPs", flops, epoch)
        tfboard_writer.add_scalar("train/Params", params, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            best_acc1 = acc1
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=args.tmp)

        logger.info("Best acc1=%.5f" % best_acc1)

        model_sparsity = get_sparsity(get_factors(model), thres=args.sparse_thres)


        target_sparsity = args.percent
        sparsity_gain = (model_sparsity - last_sparsity)
        expected_sparsity_gain = (target_sparsity - model_sparsity) / (args.epochs - epoch)

        # not sparse enough
        if model_sparsity < target_sparsity:
            if sparsity_gain < expected_sparsity_gain:
                logger.info("Sparsity gain %f (expected%f), increasing sparse penalty."%(sparsity_gain, expected_sparsity_gain))
                args.sparsity += args.delta_lambda
        # over sparse
        elif model_sparsity > target_sparsity:
            if model_sparsity > last_sparsity and args.sparsity > 0:
                args.sparsity -= args.delta_lambda
        # minimal sparsity=0
        args.sparsity = max(args.sparsity, 0)

        logger.info("Model sparsity=%f (last=%f, target=%f), args.sparsity=%f" %\
            (model_sparsity, last_sparsity, target_sparsity, args.sparsity))

        last_sparsity = model_sparsity

        lr = optimizer.param_groups[0]["lr"]
        bn_l1 = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_l1 += m.weight.abs().mean()

        tfboard_writer.add_scalar('train/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('train/lr_epoch', lr, epoch)
        tfboard_writer.add_scalar('train/BN-L1', bn_l1, epoch)
        tfboard_writer.add_scalar('train/model sparsity', model_sparsity, epoch)
        tfboard_writer.add_scalar('train/sparse penalty', args.sparsity, epoch)

        tfboard_writer.add_scalar('test/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('test/acc5_epoch', acc5, epoch)

        for k, v in get_factors(model).items():
            tfboard_writer.add_histogram("train/%s"%k, v.cpu().numpy(), epoch)

    logger.info("Optimization done, ALL results saved to %s." % args.tmp)

    # evaluate before pruning
    logger.info("evaluating before pruning...")
    acc1, acc5 = validate(val_loader, model, args.epochs)
    tfboard_writer.add_scalar('retrain/acc1_epoch', acc1, -2)
    tfboard_writer.add_scalar('retrain/acc5_epoch', acc5, -2)

    # calculate pruning mask and perform mask pruning
    model = model.module
    factors = get_factors(model)
    # mask pruning

    thres = get_threshold(model, args.percent, get_sparsity_from_model)
    logger.info("Prune rate %.3E, threshold %.3E" % (args.percent, thres))
    prune_mask = mask_prune(model, factors, thres)

    # warp back to nn.DataParallel
    model = nn.DataParallel(model.cuda())

    #prune_rate = float(pruned_filters)/total_filters
    #logger.info("Totally %d filters, %d has been pruned, pruning rate %f"%(total_filters, pruned_filters, prune_rate))

    logger.info("evaluating after masking...")
    validate(val_loader, model, args.epochs)

    # reload model
    del model
    model = eval(model_name).cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(join(args.tmp, "checkpoint.pth"))["state_dict"])

    # do real pruning
    # the Bottleneck architecture is: BN1 -> select -> conv1 -> BN2 -> conv2 -> BN3 -> conv3
    model = model.module
    model = real_prune(model, prune_mask)
    model = model.cuda()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).cuda(), ), verbose=False)
    logger.info("FLOPs %.3E, Params %.3E (after real pruning)" % (flops, params))

    bn_l1 = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_l1 += m.weight.abs().mean()
    logger.info("BN-L1 after real pruning: %f"%bn_l1.item())

    # warp back to nn.DataParallel
    model = nn.DataParallel(model)
    logger.info("evaluating after real pruning...")
    acc1, acc5 = validate(val_loader, model, args.epochs)


    if args.rescalebn:
        model = model.module
        for name, m in model.named_modules():
            if isinstance(m, Bottleneck):

                f = m.bn2.weight.data.abs()
                assert f.gt(0).all(), f
                m.bn2.weight.data.fill_(1)
                m.bn2.bias.data /= f
                f = f.view(1, -1, 1, 1)
                m.conv2.weight.data *= f

                f = m.bn3.weight.data.abs()
                assert f.gt(0).all(), f
                m.bn3.weight.data.fill_(1)
                m.bn3.bias.data /= f
                f = f.view(1, -1, 1, 1)
                m.conv3.weight.data *= f
        bn_l1 = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_l1 += m.weight.abs().mean()
        logger.info("BN-L1 after bn-rescale: %f"%bn_l1.item())

        model = nn.DataParallel(model)
        logger.info("evaluating after bn rescale...")
        acc1, acc5 = validate(val_loader, model, args.epochs)
        logger.info("Accuracy top1 %.4f, top5 %.4f" % (acc1, acc5))


    # shutdown when `args.no-retrain` is triggered
    if args.no_retrain: return

    bn_l1 = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_l1 += m.weight.abs().mean()
    tfboard_writer.add_scalar('retrain/BN-L1', bn_l1, -1)
    tfboard_writer.add_scalar('retrain/acc1_epoch', acc1, -1)
    tfboard_writer.add_scalar('retrain/acc5_epoch', acc5, -1)

    # retrain
    optimizer_retrain = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    scheduler_retrain = MultiStepLR(optimizer_retrain, milestones=milestones, gamma=args.gamma)

    best_acc1 = 0
    for epoch in range(0, args.retrain_epochs):

        # train and evaluate
        loss = train(train_loader, model, optimizer_retrain, epoch)
        acc1, acc5 = validate(val_loader, model, epoch)
        scheduler_retrain.step()

        lr = optimizer_retrain.param_groups[0]["lr"]

        bn_l1 = 0
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_l1 += m.weight.abs().mean()

        tfboard_writer.add_scalar('retrain/loss_epoch', loss, epoch)
        tfboard_writer.add_scalar('retrain/BN-L1', bn_l1, epoch)
        tfboard_writer.add_scalar('retrain/lr_epoch', lr, epoch)
        tfboard_writer.add_scalar('retrain/acc1_epoch', acc1, epoch)
        tfboard_writer.add_scalar('retrain/acc5_epoch', acc5, epoch)

        # remember best prec@1 and save checkpoint
        is_best = acc1 > best_acc1

        if is_best:
            best_acc1 = acc1

        logger.info("Best acc1=%.5f" % best_acc1)

        # save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict()
            }, is_best, path=args.tmp, filename="checkpoint-retrain0.pth")

    logger.info("Retrain done, results are saved to %s. Best acc1=%.6f, FLOPs=%.4E, Params=%.4E" % \
                  (args.tmp, best_acc1, flops, params))

def train(train_loader, model, optimizer, epoch, l1lambda=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)

        output = model(data)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # impose L1 penalty to BN factors
        if l1lambda != 0:
            for m in model.modules():
                if isinstance(m, Bottleneck):
                    m.bn1.weight.grad.add_(l1lambda*torch.sign(m.bn1.weight.data))
                    m.bn2.weight.grad.add_(l1lambda*torch.sign(m.bn2.weight.data))
                    m.bn3.weight.grad.add_(l1lambda*torch.sign(m.bn3.weight.data))

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]

        if i % args.print_freq == 0:
            logger.info('Epoch[{0}/{1}] Iter[{2}/{3}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Train Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'LR {lr:.3E} L1 {l1:.2E}'.format(
                   epoch, args.epochs, i, len(train_loader),
                   batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5,
                   lr=lr, l1=l1lambda))

    return losses.avg

def validate(val_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(val_loader):
            
            target = target.cuda(non_blocking=True)
            data = data.cuda()
            # compute output
            output = model(data)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info('Test: [{0}/{1}]\t'
                      'Test Loss {loss.val:.3f} (avg={loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} (avg={top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} (avg={top5.avg:.3f})'.format(
                       i, len(val_loader), loss=losses, top1=top1, top5=top5))

        logger.info(' * Prec@1 {top1.avg:.5f} Prec@5 {top5.avg:.5f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
