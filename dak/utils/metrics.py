import torch
import torch.nn.functional as F
import gpytorch


class RegressionMetrics:
    def __init__(self, pred_mean, pred_var, test_y) -> None:
        self.pred_mean = pred_mean  # [n_test]-size tensor
        self.pred_var = pred_var  # [n_test]-size tensor
        self.test_y = test_y  # [n_test]-size tensor
        self.metrics = {}

    def rmse(self):
        pred_mean = self.pred_mean  # [n_test]-size tensor
        test_y = self.test_y  # [n_test]-size tensor

        rmse_val = ((test_y - pred_mean).detach() ** 2).mean().sqrt().item()  # a number
        self.metrics['rmse'] = rmse_val
        return rmse_val

    def nlpd(self):
        pred_mean = self.pred_mean  # [n_test]-size tensor
        pred_var = self.pred_var  # [n_test]-size tensor
        test_y = self.test_y  # [n_test]-size tensor

        model_eval_at_test_x = gpytorch.distributions.MultivariateNormal(pred_mean, pred_var.diag(0))
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.test_y.device)
        trained_pred_dist = likelihood(model_eval_at_test_x)

        nlpd_val = gpytorch.metrics.negative_log_predictive_density(trained_pred_dist, test_y).item()  # a number
        self.metrics['nlpd'] = nlpd_val

        return nlpd_val

    def coverage_score(self, num_std=1):
        pred_mean = self.pred_mean  # [n_test]-size tensor
        pred_var = self.pred_var  # [n_test]-size tensor
        test_y = self.test_y  # [n_test]-size tensor

        pred_lower = pred_mean - num_std * pred_var.sqrt()
        pred_upper = pred_mean + num_std * pred_var.sqrt()
        coverage_bool = (test_y >= pred_lower) & (test_y <= pred_upper)

        coverage_score_val = (coverage_bool.sum() / len(coverage_bool)).item()

        self.metrics['coverage_score'] = coverage_score_val

        return coverage_score_val


class ClassificationMetrics:
    def __init__(self, num_mc=20, n_bins=15, option='logits'):
        self.num_mc = num_mc
        self.n_bins = n_bins
        self.option = option
        assert self.option in ['logits', 'probs']

    def accuracy(self, model, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(model.device), target.to(model.device)
                output_ = []
                for mc_run in range(self.num_mc):
                    output, _ = model(data)
                    output_.append(output)
                output = torch.mean(torch.stack(output_), dim=0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += data.size(0)
        acc = 100. * correct / total
        return acc

    def nll(self, model, test_loader):
        nll = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(model.device), target.to(model.device)
                output_ = []
                for mc_run in range(self.num_mc):
                    output, _ = model(data)
                    output_.append(output)
                output = torch.mean(torch.stack(output_), dim=0)
                if self.option == 'logits':
                    nll += F.cross_entropy(output, target, reduction='sum').item()
                elif self.option == 'probs':
                    nll += F.nll_loss(torch.log(output), target, reduction='sum').item()
        nll /= len(test_loader.dataset)
        return nll

    def ece(self, model, test_loader):
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(model.device), target.to(model.device)
                output_ = []
                for mc_run in range(self.num_mc):
                    output, _ = model(data)
                    output_.append(output)
                output = torch.mean(torch.stack(output_), dim=0)
                if self.option == 'logits':
                    all_probs.append(F.softmax(output, dim=1))
                elif self.option == 'probs':
                    all_probs.append(output)
                all_labels.append(target)

        probs = torch.cat(all_probs)
        labels = torch.cat(all_labels)

        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences, predictions = torch.max(probs, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=probs.device)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece.item()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
