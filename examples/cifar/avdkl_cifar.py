import os
import sys
from pathlib import Path  # if you haven't already done so

file = Path(os.path.dirname(os.path.abspath(__file__))).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import time
import argparse
import torch.backends.cudnn as cudnn


import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import gpytorch
import math

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import gpinfuser
import dak.models.deterministic.resnet_large as resnet
from dak.utils.metrics import AverageMeter
from dak.utils.util import accuracy, ece_score


def get_mean():
    return gpytorch.means.ConstantMean()

def get_kernel(num_features):
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    kernel.initialize(**{'base_kernel.lengthscale': 15})
    return kernel

def get_likelihood(num_features, num_classes):
    likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, num_classes=num_classes),
    return likelihood

def get_amortized_svgp(num_inducing, num_features):
    return gpinfuser.models.AmortizedSVGP(get_mean(), get_kernel(num_features), num_inducing, 1)

def get_feature_extractor():
    return resnet.resnet18(num_classes=10, classifier=False)

def get_variational_module(in_features, num_features, num_inducing, saturation):
    return gpinfuser.nn.Variational(
        in_features=in_features,
        num_tasks=10,
        num_inducing=num_inducing,
        num_features=num_features,
        saturation=saturation
    )

def get_amortized_variational_dkl(num_features, num_inducing, nonlinearity, saturation):
    feature_extractor = get_feature_extractor()
    variational_module = get_variational_module(num_features, num_features, num_inducing, saturation)
    gplayer = get_amortized_svgp(num_inducing, num_features)
    return gpinfuser.models.AVDKL(feature_extractor, variational_module, gplayer, get_likelihood(num_features, 10))

def get_optimizer(model, lr=0.1, weight_decay=1e-4):
    return torch.optim.SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': weight_decay},
        {'params': model.variational_estimator.parameters(), 'weight_decay': weight_decay},
        {'params': model.gplayer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.likelihood.parameters()}
    ], lr=lr, momentum=0.9, weight_decay=0)

def get_scheduler(optimizer, epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def warmup_lr_lambda(epoch):
    if epoch < 5:  # Assuming a 5-epoch warm-up period
        return epoch / 5
    else:
        return 1


class AVDKLcifar:
    def __init__(self, args):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        print("Using: ", self.device)

        torch.manual_seed(args.seed)

        self.epochs = args.epochs
        self.log_interval = args.log_interval
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.num_mc_train = args.num_mc_train
        self.num_mc_test = args.num_mc_test

        num_features = args.num_classes
        num_inducing = 64

        # feature_extractor = resnet.resnet18(num_classes=10, classifier=False)
        feature_extractor = resnet.__dict__[args.arch](num_classes=num_features, classifier=True)
        variational_module = gpinfuser.nn.Variational(
            in_features=num_features,
            num_tasks=num_features,
            num_inducing=num_inducing,
            num_features=num_features,
        )
        gplayer = get_amortized_svgp(num_inducing, num_features)

        likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_features, num_classes=args.num_classes).to(self.device)
        self.model = gpinfuser.models.AVDKL(feature_extractor, variational_module, gplayer, likelihood).to(self.device)

        if args.pretrained:
            self.model.feature_extractor.load_state_dict(
                torch.load('./checkpoint/nn_cifar_100_batch_128.pth')['state_dict']
            )

        self.optimizer = get_optimizer(self.model, lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.warmup_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_lr_lambda)

        if args.fix_features and args.pretrained:
            for name, params in self.model.named_parameters():
                if "feature_extractor" in name:
                    params.requires_grad = False

    def train(self, train_loader, epoch):
        self.model.train()
        self.model.likelihood.train()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        mll = gpytorch.mlls.VariationalELBO(self.model.likelihood, self.model.gplayer,
                                            num_data=len(train_loader.dataset))
        with gpytorch.settings.num_likelihood_samples(self.num_mc_train):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = -mll(output, target)
                loss.backward(torch.ones_like(loss))
                self.optimizer.step()

                # output_dist = self.model.likelihood(output)
                # output_probs = output_dist.mean.squeeze()
                # prec1 = accuracy(output_probs, target)[0]

                output_probs = self.model.likelihood(output).probs.mean(0).float()
                prec1 = accuracy(output_probs.data, target)[0]

                loss = loss.float().mean()
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))

                if batch_idx % self.log_interval == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch,
                        batch_idx,
                        len(train_loader),
                        loss=losses,
                        top1=top1))
        if epoch <= 5:
            self.warmup_scheduler.step()

        return losses.avg, top1.avg

    def test(self, test_loader, ece_bins=10):
        self.model.eval()
        self.model.likelihood.eval()

        correct = 0
        nll = 0
        ece = 0
        batch_count = 0
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(self.num_mc_test):
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model.likelihood(self.model(data))  # samples from the predictive distribution

                output_probs = output.probs.mean(0)
                pred = output_probs.argmax(-1)

                correct += pred.eq(target.view_as(pred)).sum().item()
                nll += F.nll_loss(torch.log(output.probs.mean(0)), target, reduction='sum').item()
                ece += ece_score(output_probs.cpu().numpy(), target.cpu().numpy(), n_bins=ece_bins)
                batch_count += 1

        acc = 100. * correct / len(test_loader.dataset)
        nll /= len(test_loader.dataset)
        ece /= batch_count

        return acc, nll, ece

    def validate(self, val_loader):
        self.model.eval()
        self.model.likelihood.eval()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        nll = AverageMeter('NLL', ':.4e')
        mll = gpytorch.mlls.VariationalELBO(self.model.likelihood, self.model.gplayer,
                                            num_data=len(val_loader.dataset))
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss = -mll(output, target)

                output_probs = self.model.likelihood(output).probs.mean(0).float()
                loss = val_loss.float().mean()
                neg_log_likelihood = F.nll_loss(torch.log(output_probs), target)

                # measure accuracy and record loss
                prec1 = accuracy(output_probs.data, target)[0]
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
                nll.update(neg_log_likelihood.item(), data.size(0))

        print(
            '\nValidation set: Average loss: {:.4f},  Prec@1: {}/{} ({:.2f}%)\n'.format(
                losses.avg, top1.sum / 100, len(val_loader.dataset),
                top1.avg))

        return top1.avg, nll.avg


model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and name.startswith("resnet") and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='CIFAR')
parser.add_argument('--mode',
                    default='test',
                    type=str,
                    choices=['train', 'test', 'val'],
                    help='train | test')
parser.add_argument('--model',
                    type=str,
                    default='avdkl',
                    choices=['nn', 'nnsvgp', 'svdkl', 'dak'],
                    help='Choose the DKL models to use.')
parser.add_argument('--arch',
                    '-a',
                    metavar='ARCH',
                    default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--num_classes',
                    type=int,
                    default=10,
                    choices=[10, 100],
                    help='Number of classes in the dataset (default: 10 for CIFAR-10)')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--seed',
                    type=int,
                    default=10,
                    metavar='S',
                    help='random seed (default: 10)')
parser.add_argument('-pretrained',
                    default=False,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if use pre-trained NN; (default: False)')
parser.add_argument('-fix-features',
                    default=False,
                    type=bool,
                    action=argparse.BooleanOptionalAction,
                    help='if use pre-trained NN; (default: False)')
parser.add_argument('--epochs',
                    default=200,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch-size',
                    default=1024,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('--test-batch-size',
                    type=int,
                    default=1000,
                    metavar='N',
                    help='input batch size for testing (default: 10000)')
parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')
parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gamma',
                    type=float,
                    default=0.2,
                    metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--num_mc_test',
                    type=int,
                    default=20,
                    metavar='N',
                    help='number of Monte Carlo samples to be drawn during inference')
parser.add_argument('--num_mc_train',
                    type=int,
                    default=8,
                    metavar='N',
                    help='number of Monte Carlo runs during training')
parser.add_argument('--num-proj',
                    default=64,
                    type=int,
                    metavar='N',
                    help='number of projections (base GP layers) (default: 16)')
parser.add_argument('--num-svdkl-proj',
                    default=64,
                    type=int,
                    metavar='N',
                    help='number of projections (base GP layers) for SVDKL model (default: 10)')
parser.add_argument('--num-nnsvgp-proj',
                    default=64,
                    type=int,
                    metavar='N',
                    help='number of output features of NN for NNSVGP model (default: 10)')
parser.add_argument('--num-ip',
                    default=64,
                    type=int,
                    metavar='N',
                    help='number of inducing points for SVGP (default: 64)')
parser.add_argument('--num-ip-svgp',
                    default=512,
                    type=int,
                    metavar='N',
                    help='number of inducing points for SVGP (default: 64)')
parser.add_argument('--noise-var',
                    default=0.1,
                    type=float,
                    metavar='NV',
                    help='noise variance (default: 0.1)')
parser.add_argument('--log-interval',
                    type=int,
                    default=195,
                    metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint-dir',
                    type=str,
                    default='./checkpoint')
parser.add_argument('--log-dir',
                    type=str,
                    default='./logs')
parser.add_argument('--resume',
                    default='',
                    type=str,
                    metavar='PATH',
                    help='path to latest checkpoint (default: none)')

best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    ###################################
    # Model Setup
    ###################################
    assert args.model in ['avdkl']
    cifar = AVDKLcifar(args)

    ###################################
    # optionally resume from a checkpoint
    ###################################
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            cifar.model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    ###################################
    # Data Loading
    ###################################
    assert args.num_classes in [10, 100]
    if args.num_classes == 10:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2023, 0.1994, 0.2010])
        train_set = datasets.CIFAR10(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
        test_set = datasets.CIFAR10(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
    else:
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                         std=[0.2675, 0.2565, 0.2761])
        train_set = datasets.CIFAR100(
            root='./data',
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
        test_set = datasets.CIFAR100(
            root='./data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    ###################################
    # Log directory
    ###################################
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    ###################################
    # Training and Testing
    ###################################
    print(args.mode)
    start = time.time()
    if args.mode == 'train':
        log_dir = args.log_dir
        current_num_files = next(os.walk(log_dir))[2]  # get all files in the directory
        run_num = len(current_num_files)
        log_f_name = log_dir + '/CIFAR' + str(args.num_classes) + "_" + args.model + "_batch_" + str(args.batch_size) + "_log_" + str(run_num) + ".csv"
        print("logging at : " + log_f_name)

        # logging file
        log_f = open(log_f_name, "w+")
        log_f.write('epoch,loss,acc,top1,nll\n')

        for epoch in range(args.epochs):
            if epoch > 5:
                cifar.scheduler.step()

            if args.resume:
                if epoch < args.start_epoch:
                    continue

            loss, top1 = cifar.train(train_loader, epoch)
            prec1, nll = cifar.validate(test_loader)
            # cifar.scheduler.step()

            log_f.write('{},{},{},{},{}\n'.format(epoch, loss, top1, prec1, nll))
            log_f.flush()

            batch_time = time.time() - start
            print('Runtime: ' + str(batch_time))

            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            if is_best:
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'state_dict': cifar.model.state_dict(),
                        'best_prec1': best_prec1,
                    },
                    os.path.join(
                        args.checkpoint_dir,
                        '{}_cifar_{}_batch_{}.pth'.format(args.model,
                                                           args.num_classes,
                                                           args.batch_size)
                    )
                )

        log_f.close()

    elif args.mode == 'test':
        checkpoint_file = args.checkpoint_dir + '/{}_cifar_{}_batch_{}.pth'.format(args.model,
                                                                                   args.num_classes,
                                                                                   args.batch_size)
        checkpoint = torch.load(checkpoint_file)
        cifar.model.load_state_dict(checkpoint['state_dict'])

        acc, nll, ece = cifar.test(test_loader)
        print("Accuracy: ", acc)
        print("NLL: ", nll)
        print("ECE: ", ece)
    else:
        checkpoint_file = args.checkpoint_dir + '/{}_cifar_{}_batch_{}.pth'.format(args.model,
                                                                                   args.num_classes,
                                                                                   args.batch_size)
        checkpoint = torch.load(checkpoint_file)
        cifar.model.load_state_dict(checkpoint['state_dict'])
        top1, _ = cifar.validate(test_loader)
        print(' * Prec@1 {top1:.3f}'.format(top1=top1))

    end = time.time()
    print("Done. Total time: " + str(end - start))
    print("Average time per epoch: " + str((end - start) / args.epochs))


if __name__ == '__main__':
    main()

    ## Results: CIFAR-10
    ## batch size 128: 77.61, 1.795, 0.164; 77.34, 1.779, 0.167
    ## batch size 1024: 77.0, 2.305, 0.175; 77.15, 2.348, 0.176

    ## Results: CIFAR-100
    ## batch size 128: 93.77, 0.411, 0.052; 94.69, 0.292, 0.043
    ## batch size 1024: 93.23, 0.424, 0.053; 93.42, 0.455, 0.054