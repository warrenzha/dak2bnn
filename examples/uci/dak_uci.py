import os
import torch
from tqdm import tqdm

from dak.models.deterministic import resnet1d, fnn
from dak.models.dak_variational import DAK


class DAKUCI(object):
    def __init__(self, args, mc_sampling=False, nn_out_features=16, batch_size=128) -> None:
        self.args = args
        self.mc_sampling = mc_sampling
        self.nn_out_features = nn_out_features
        self.batch_size = batch_size

    def model_setup(self):
        args = self.args
        ###################################
        # Define Model
        ###################################
        if args.dnn_name == 'resnet1d':
            feature_extractor = resnet1d.__dict__['resnet1d20'](num_features=self.nn_out_features)
        elif args.dnn_name == 'fnn':
            feature_extractor = fnn.FNNRegression(
                num_features=self.nn_out_features,
                hidden_features=args.hidden_features,
            )

        model = DAK(
            feature_extractor=feature_extractor,
            num_classes=1,
            num_features=args.num_proj,
            inducing_level=3,
            grid_bounds=(0, 1),  #(-0.5,0.5),#(-0.25, 1.25),
        )
        # model = torch.compile(model,mode='default') # Compile it
        criterion = torch.nn.GaussianNLLLoss()  # torch.nn.MSELoss()

        # enable cuda
        if torch.cuda.is_available():
            model.cuda()
            criterion.cuda()
        if args.half:
            model.half()
            criterion.half()

        self.model = model
        self.criterion = criterion

    def train(self, train_loader, val_loader=None):

        args = self.args
        model = self.model
        criterion = self.criterion

        # switch to train mode
        model.train()

        # train
        train_losses = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
        epochs_iter = tqdm(range(args.epochs), desc=f"Training {model.__class__.__qualname__}")
        minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
        for epoch in epochs_iter:
            running_loss = 0.0

            for (input, target) in minibatch_iter:

                # enable cuda
                input, target = input.to(args.device), target.to(args.device)

                if self.mc_sampling:
                    ############################################
                    # Compute NLL by MC sampling
                    ############################################
                    outputs, kls = model(input, num_mc=args.num_mc_train,
                                         return_kl=True, return_sampling=True)
                    # outputs id of [num_mc_train, len(target)] size
                    output_mean = torch.mean(outputs, dim=0)
                    kl = torch.mean(kls, dim=0)
                    nll = criterion(
                        outputs, target, args.noise_var ** 2 * torch.ones_like(outputs),
                    ).mean(dim=0)
                else:
                    ############################################
                    # Compute NLL by closed form
                    ############################################
                    mean_var_output, kl = model(input,
                                                return_kl=True, return_sampling=False)
                    output_mean = mean_var_output.mean
                    output_var = mean_var_output.var

                    nll = criterion(
                        output_mean, target, args.noise_var ** 2 * torch.ones_like(output_mean),
                    ) + output_var.sum()

                scaled_kl = kl / self.batch_size
                loss = nll + scaled_kl  # ELBO loss

                # compute gradient and do SGD step
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                output_mean = output_mean.float()
                loss = loss.float()

                # record loss
                running_loss += loss.item() * input.size(0) / self.batch_size

            epochs_iter.set_postfix(loss=loss.item() / self.batch_size)
            train_loss = running_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            # validate
            if args.validate:
                best_loss = float('inf')
                val_loss = self.validate(args, val_loader, model,
                                         criterion, self.mc_sampling, self.batch_size,
                                         )
                if args.verbose:
                    print(f'Epoch {epoch + 1}\t\t',
                          f'Training Loss: {train_loss}\t\t',
                          f'Validation Loss: {val_loss}')
                if val_loss < best_loss:
                    best_loss = val_loss
                    filename = f'{model.__class__.__qualname__.lower()}_{args.dnn_name}_{train_loader.dataset.dataset.__name__}.pth'
                    save_path = os.path.join(args.checkpoint_dir, filename)
                    torch.save(model.state_dict(), save_path)

        self.train_loader = train_loader
        self.model = model
        self.train_losses = train_losses
        return model, train_losses

    @staticmethod
    def validate(args, val_loader, model, criterion, mc_sampling, batch_size):
        # switch to evaluate mode
        model.eval()

        # validate
        with torch.no_grad():
            running_loss = 0.0
            for (input, target) in enumerate(val_loader):

                # enable cuda
                input, target = input.to(args.device), target.to(args.device)

                if mc_sampling:
                    ############################################
                    # Compute NLL by MC sampling
                    ############################################
                    outputs, kls = model(input, num_mc=args.num_mc_train, return_kl=True, return_sampling=True)
                    # outputs is of [num_mc_train, len(target)] size
                    output_mean = torch.mean(outputs, dim=0)
                    kl = torch.mean(kls, dim=0)

                    nll = criterion(
                        outputs, target, args.noise_var * torch.ones_like(outputs),
                    ).mean(dim=0)
                else:
                    ############################################
                    # Compute NLL by closed form
                    ############################################
                    mean_var_output, kl = model(input, return_kl=True, return_sampling=False)
                    output_mean = mean_var_output.mean
                    output_var = mean_var_output.var

                    nll = criterion(
                        output_mean, target, args.noise_var * torch.ones_like(output_mean),
                    ) + output_var.sum()

                scaled_kl = kl / batch_size
                loss = nll + scaled_kl  # ELBO loss

                output_mean = output_mean.float()
                loss = loss.float()

                # record loss
                running_loss += loss.item() * input.size(0) / batch_size
            val_loss = running_loss / len(val_loader.dataset)
        return val_loss

    def test(self, test_loader, model):

        args = self.args

        # initialize
        means_list = []
        stds_list = []
        input_list = []
        target_list = []

        # switch to evaluate mode
        model.eval()

        # test
        with torch.no_grad():
            for (input, target) in test_loader:
                # enable cuda
                input, target = input.to(args.device), target.to(args.device)

                if self.mc_sampling:
                    ############################################
                    # Compute NLL by MC sampling
                    ############################################

                    # Compute output
                    outputs, _ = model(input, num_mc=args.num_mc_test, return_kl=True, return_sampling=True)
                    means = outputs.mean(dim=0)
                    stds = outputs.std(dim=0)
                else:
                    ############################################
                    # get mean and variance by closed form
                    ############################################
                    mean_var_output, _ = model(input, return_kl=True, return_sampling=False)
                    means = mean_var_output.mean
                    stds = mean_var_output.var.sqrt()
                means_list.append(means)
                stds_list.append(stds)
                input_list.append(input)
                target_list.append(target)

            # get mean and std
            pred_mean = torch.cat(means_list)
            pred_std = torch.cat(stds_list)
            input_true = torch.cat(input_list)
            target_true = torch.cat(target_list)

        self.pred_mean = pred_mean
        self.pred_std = pred_std
        self.input_true = input_true
        self.target_true = target_true
        return pred_mean, pred_std, input_true, target_true
