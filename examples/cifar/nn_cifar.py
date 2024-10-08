import torch
import torch.nn.functional as F

import dak.models.deterministic.resnet as resnet
import dak.models.deterministic.resnet_large as resnet_large
from dak.utils.util import ece_score, accuracy
from dak.utils.metrics import AverageMeter


class NNCIFAR:
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

        if args.num_classes == 10:
            model = resnet_large.__dict__[args.arch](num_classes=args.num_classes, classifier=True)
        else:
            model = resnet_large.__dict__[args.arch](num_classes=args.num_classes, classifier=True)

        self.model = model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epochs)

    def train(self, train_loader, epoch):
        self.model.train()

        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)

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
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch,
                                                                      batch_idx,
                                                                      len(train_loader),
                                                                      loss=losses,
                                                                      top1=top1))

        return losses.avg, top1.avg

    def test(self, test_loader, ece_bins=10):
        self.model.eval()
        correct = 0
        nll = 0
        ece = 0
        batch_count = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
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

    def validate(self, val_loader, ece_bins=10):
        self.model.eval()
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')

        with torch.no_grad():
            for i, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss = F.cross_entropy(output, target)

                output = output.float()
                loss = val_loss.float()
                neg_log_likelihood = F.cross_entropy(output, target)

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))

        print(
            '\nValidation set: Average loss: {:.4f},  Prec@1: {}/{} ({:.2f}%)\n'.format(
                losses.avg, top1.sum / 100, len(val_loader.dataset),
                top1.avg))

        return top1.avg, losses.avg


if __name__ == '__main__':
    resnet_18_model = resnet_large.resnet18(num_classes=10, classifier=True)
    resnet_34_model = resnet_large.resnet34(num_classes=100, classifier=True)

    total_resnet18_params = sum(p.numel() for p in resnet_18_model.parameters())
    print("Total resnet 18 parameters: ", total_resnet18_params)

    total_resnet34_params = sum(p.numel() for p in resnet_34_model.parameters())
    print("Total resnet 34 parameters: ", total_resnet34_params)