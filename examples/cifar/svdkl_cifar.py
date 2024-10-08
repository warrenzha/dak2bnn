import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import gpytorch

import dak.models.deterministic.resnet_large as resnet_large
from dak.models.svdkl_variational import SVDKL
from dak.utils.util import ece_score, accuracy
from dak.utils.metrics import AverageMeter


def warmup_lr_lambda(epoch):
    if epoch < 5:  # Assuming a 5-epoch warm-up period
        return epoch / 5
    else:
        return 1


class SVDKLCIFAR:
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

        if args.num_classes == 10:
            feature_extractor = resnet_large.__dict__[args.arch](num_classes=args.num_classes, classifier=False)
            num_dim = args.num_svdkl_proj
            model = SVDKL(
                feature_extractor=feature_extractor,
                num_dim=num_dim,
                grid_bounds=(-1., 1.),
                grid_size=args.num_ip,
                likelihood=gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_dim,
                                                                  num_classes=args.num_classes),
                embedding=torch.nn.Linear(feature_extractor.fc.in_features, num_dim)
            )
            if args.pretrained:
                model.feature_extractor.load_state_dict(
                    torch.load('./checkpoint/nn_cifar_10_batch_128.pth')['state_dict']
                )
        else:
            feature_extractor = resnet_large.__dict__[args.arch](num_classes=args.num_classes, classifier=False)
            num_dim = 512
            model = SVDKL(
                feature_extractor=feature_extractor,
                num_dim=num_dim,
                grid_bounds=(-1., 1.),
                grid_size=args.num_ip,
                likelihood=gpytorch.likelihoods.SoftmaxLikelihood(num_features=num_dim,
                                                                  num_classes=args.num_classes),
            )
            if args.pretrained:
                model.feature_extractor.load_state_dict(
                    torch.load('./checkpoint/nn_cifar_100_batch_128.pth')['state_dict']
                )

        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        self.warmup_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_lr_lambda)

        if args.fix_features and args.pretrained:
            for name, params in self.model.named_parameters():
                if "feature_extractor" in name:
                    params.requires_grad = False

    def reset_optimizer(self, epoch):
        lr = self.lr * (0.1 ** (epoch // 40))
        self.optimizer = torch.optim.SGD([
            {'params': self.model.feature_extractor.parameters(), 'weight_decay': self.weight_decay},
            {'params': self.model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
            {'params': self.model.gp_layer.variational_parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=lr, weight_decay=0)

    def train(self, train_loader, epoch):
        self.model.train()
        self.model.likelihood.train()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        mll = gpytorch.mlls.VariationalELBO(self.model.likelihood, self.model.gp_layer,
                                            num_data=len(train_loader.dataset))
        with gpytorch.settings.num_likelihood_samples(self.num_mc_train):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = -mll(output, target)
                loss.backward()
                self.optimizer.step()

                output_probs = self.model.likelihood(output).probs.mean(0).float()
                loss = loss.float()
                # measure accuracy and record loss
                prec1 = accuracy(output_probs.data, target)[0]
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
        mll = gpytorch.mlls.VariationalELBO(self.model.likelihood, self.model.gp_layer,
                                            num_data=len(val_loader.dataset))
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss = -mll(output, target)

                output_probs = self.model.likelihood(output).probs.mean(0).float()
                loss = val_loss.float()
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


if __name__ == '__main__':
    resnet_18_model = resnet_large.resnet18(num_classes=64, classifier=True)
    resnet_34_model = resnet_large.resnet34(num_classes=100, classifier=False)

    model_10 = SVDKL(
        feature_extractor=resnet_18_model,
        num_dim=64,
        grid_size=64,
        likelihood=gpytorch.likelihoods.SoftmaxLikelihood(num_features=64,
                                                          num_classes=10),
    )

    model_100 = SVDKL(
        feature_extractor=resnet_34_model,
        num_dim=512,
        grid_size=64,
        likelihood=gpytorch.likelihoods.SoftmaxLikelihood(num_features=512,
                                                          num_classes=100),
    )

    total_resnet18_params = sum(p.numel() for p in model_10.parameters())
    print("Total resnet 18 parameters: ", total_resnet18_params)

    total_resnet34_params = sum(p.numel() for p in model_100.parameters())
    print("Total resnet 34 parameters: ", total_resnet34_params)
