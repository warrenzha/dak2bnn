import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from dak.models.deterministic.cnn import CNN
from dak.models.dak_variational import DAK
from dak.utils.util import ece_score

class NNMNIST:
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
        self.gamma = args.gamma
        self.weight_decay = args.weight_decay
        self.num_mc_train = args.num_mc_train
        self.num_mc_test = args.num_mc_test

        feature_extractor = CNN(num_classes=10, classifier=True)
        self.model = feature_extractor.to(self.device)

        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.gamma)

    def reset_optimizer(self, epoch):
        lr = self.lr * (0.1 ** (epoch // 10))

        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=lr, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)

    def train(self, train_loader, epoch):
        self.model.train()

        losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if batch_idx % self.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

        return losses

    def test(self, test_loader, ece_bins=10):
        self.model.eval()
        correct = 0
        nll = 0
        ece = 0
        batch_count = 0
        with torch.no_grad():
            for data, target in test_loader:
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

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(val_loader.dataset)
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                val_loss, correct, len(val_loader.dataset),
                100. * correct / len(val_loader.dataset)))

        return val_loss, 100. * correct / len(val_loader.dataset)


if __name__ == '__main__':
    feature_extractor = CNN(num_classes=10, classifier=True)
    model = feature_extractor

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)