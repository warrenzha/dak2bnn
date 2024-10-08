import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import gpytorch

from dak.models.deterministic.cnn import CNN
from dak.models.svdkl_variational import SVDKL
from dak.utils.util import ece_score


class SVDKLMNIST:
    def __init__(self, args):
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        torch.manual_seed(args.seed)

        self.epochs = args.epochs
        self.log_interval = args.log_interval
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.gamma = args.gamma
        self.weight_decay = args.weight_decay
        self.num_mc_train = args.num_mc_train
        self.num_mc_test = args.num_mc_test

        feature_extractor = CNN(num_classes=args.num_svdkl_proj, classifier=True)
        self.model = SVDKL(
            feature_extractor=feature_extractor,
            num_dim=feature_extractor.out_features,
            grid_size=args.num_ip,
            likelihood=gpytorch.likelihoods.SoftmaxLikelihood(num_features=feature_extractor.out_features,
                                                              num_classes=10)
        ).to(self.device)

        self.optimizer = torch.optim.Adadelta([
            {'params': self.model.feature_extractor.parameters(), 'weight_decay': args.weight_decay},
            {'params': self.model.gp_layer.hyperparameters(), 'lr': args.lr * 0.01},
            {'params': self.model.gp_layer.variational_parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=args.lr, weight_decay=0)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=args.gamma)

    def reset_optimizer(self, epoch):
        lr = self.lr * (0.1 ** (epoch // 10))

        self.optimizer = torch.optim.Adadelta([
            {'params': self.model.feature_extractor.parameters(), 'weight_decay': self.weight_decay},
            {'params': self.model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
            {'params': self.model.gp_layer.variational_parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=lr, weight_decay=self.weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.gamma)

    def train(self, train_loader, epoch):
        self.model.train()
        self.model.likelihood.train()
        self.reset_optimizer(epoch)

        losses = []
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
                losses.append(loss.item())

                if batch_idx % self.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.item()
                    ))

        return losses

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
                probs = output.probs.mean(0)
                pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all the sample we've drawn

                correct += pred.eq(target.view_as(pred)).cpu().sum().item()
                nll += F.nll_loss(torch.log(probs), target, reduction='sum').item()
                ece += ece_score(probs.cpu().numpy(), target.cpu().numpy(), n_bins=ece_bins)
                batch_count += 1

        acc = 100. * correct / len(test_loader.dataset)
        nll /= len(test_loader.dataset)
        ece /= batch_count

        return acc, nll, ece

    def validate(self, val_loader):
        self.model.eval()
        self.model.likelihood.eval()

        correct = 0
        val_loss = 0
        batch_count = 0
        mll = gpytorch.mlls.VariationalELBO(self.model.likelihood, self.model.gp_layer,
                                            num_data=len(val_loader.dataset))
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for batch_idx, (data, target) in enumerate(val_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                output_likelihood = self.model.likelihood(self.model(data))
                val_loss -= mll(output, target)
                pred = output_likelihood.probs.mean(0).argmax(-1)
                correct += pred.eq(target.view_as(pred)).cpu().sum()
                batch_count += 1

        val_loss /= batch_count
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                val_loss, correct, len(val_loader.dataset),
                100. * correct / len(val_loader.dataset)))

        return val_loss, 100. * correct / len(val_loader.dataset)


if __name__ == '__main__':
    feature_extractor = CNN(num_classes=16, classifier=True)
    model = SVDKL(
        feature_extractor=feature_extractor,
        num_dim=16,
        grid_size=64,
        likelihood=gpytorch.likelihoods.SoftmaxLikelihood(num_features=16,
                                                          num_classes=10)
    )

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", total_params)