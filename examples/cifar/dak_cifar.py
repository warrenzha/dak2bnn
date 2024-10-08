import torch
import torch.nn.functional as F

from dak.models.dak_variational import DAKMC
import dak.models.deterministic.resnet_large as resnet_large
from dak.utils.util import ece_score, accuracy
from dak.utils.metrics import AverageMeter


class DAKCIFAR:
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
            num_dim = args.num_proj
            model = DAKMC(
                feature_extractor=feature_extractor,
                num_classes=args.num_classes,
                num_features=num_dim,
                inducing_level=6,
                grid_bounds=(-1., 1.),
                lengthscale=1.0,
                embedding=torch.nn.Linear(feature_extractor.fc.in_features, num_dim, bias=False),
            )
            if args.pretrained:
                model.feature_extractor.load_state_dict(
                    torch.load('./checkpoint/nn_cifar_10_batch_128.pth')['state_dict']
                )
        else:
            feature_extractor = resnet_large.__dict__[args.arch](num_classes=args.num_classes, classifier=False)
            num_dim = args.num_proj
            model = DAKMC(
                feature_extractor=feature_extractor,
                num_classes=args.num_classes,
                num_features=num_dim,
                inducing_level=6,
                grid_bounds=(-1., 1.),
                lengthscale=1.0,
                embedding=torch.nn.Linear(feature_extractor.fc.in_features, num_dim, bias=False),
            )
            if args.pretrained:
                feature_extractor.load_state_dict(
                    torch.load('./checkpoint/nn_cifar_100_batch_128.pth')['state_dict']
                )

        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9,
                                         weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

        if args.fix_features and args.pretrained:
            for name, params in self.model.named_parameters():
                if "feature_extractor" in name:
                    params.requires_grad = False

    def reset_optimizer(self, epoch):
        lr = self.lr * (0.1 ** (epoch // 20))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, train_loader, epoch):
        self.model.train()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, kl = self.model(data, num_mc=self.num_mc_train)
            loss = F.cross_entropy(output, target) + kl / self.batch_size  # ELBO loss

            loss.backward()
            self.optimizer.step()

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
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

        return losses.avg, top1.avg

    def test(self, test_loader, ece_bins=10):
        self.model.eval()
        print("Length scale: ", self.model.gp_activation.kernel.lengthscale)
        correct = 0
        nll = 0
        ece = 0
        batch_count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output, _ = self.model(data, num_mc=self.num_mc_test)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item()
                nll += F.cross_entropy(output, target, reduction='sum').item()
                ece += ece_score(probs.cpu().numpy(), target.cpu().numpy(), n_bins=ece_bins)
                batch_count += 1

        acc = 100. * correct / len(test_loader.dataset)
        nll /= len(test_loader.dataset)
        ece /= batch_count

        return acc, nll, ece

    def validate(self, val_loader):
        self.model.eval()
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        nll = AverageMeter('NLL', ':.4e')
        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output, kl = self.model(data, num_mc=self.num_mc_test)
                val_loss = F.cross_entropy(output, target) + kl / 1000  # ELBO loss

                output = output.float()
                loss = val_loss.float()
                neg_log_likelihood = F.cross_entropy(output, target)

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
                nll.update(neg_log_likelihood.item(), data.size(0))

        print(
            '\nValidation set: Average loss: {:.4f},  Prec@1: {}/{} ({:.2f}%)\n'.format(
                losses.avg, top1.sum / 100, len(val_loader.dataset),
                top1.avg))

        return top1.avg, nll.avg


if __name__ == '__main__':
    resnet_18_model = resnet_large.resnet18(num_classes=10, classifier=False)
    resnet_34_model = resnet_large.resnet34(num_classes=100, classifier=False)

    model_10 = DAKMC(
        feature_extractor=resnet_18_model,
        num_classes=10,
        num_features=64,
        inducing_level=6,
        grid_bounds=(-1., 1.),
        lengthscale=1.0,
        embedding=torch.nn.Linear(resnet_18_model.fc.in_features, 64, bias=False),
    )

    model_100 = DAKMC(
        feature_extractor=resnet_34_model,
        num_classes=100,
        num_features=128,
        inducing_level=6,
        grid_bounds=(-1., 1.),
        lengthscale=1.0,
        embedding=torch.nn.Linear(resnet_34_model.fc.in_features, 128, bias=False),
    )

    total_resnet18_params = sum(p.numel() for p in model_10.parameters())
    print("Total resnet 18 parameters: ", total_resnet18_params)

    total_resnet34_params = sum(p.numel() for p in model_100.parameters())
    print("Total resnet 34 parameters: ", total_resnet34_params)
