import os
from tqdm import tqdm
import torch

import gpytorch

from dak.models.deterministic import resnet1d, fnn
from dak.models.avdkl_variational import AVDKL
class AVDKLUCI(object):
    def __init__(self, args, nn_out_features=16) -> None:
        self.args = args
        self.nn_out_features = nn_out_features
    
    def model_setup(self):
        args = self.args
        
        ###################################
        # NN
        ###################################
        if args.dnn_name == 'resnet1d':
            feature_extractor = resnet1d.__dict__['resnet1d20'](num_features=self.nn_out_features)
        elif args.dnn_name == 'fnn':
            feature_extractor = fnn.FNNRegression(
                num_features=self.nn_out_features, 
                hidden_features=args.hidden_features,
            )
        
        ###################################
        # AV-DKL
        ###################################
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = AVDKL(
            feature_extractor=feature_extractor,
            num_inducing=args.num_ip, 
            saturation=torch.nn.Tanh(),
            likelihood=likelihood,
            num_tasks=1,
        )
        # model = torch.compile(model,mode='default') # Compile it
        
        # enable cuda
        if torch.cuda.is_available():
            model.cuda()
            model.likelihood.cuda()
        if args.half:
            model.half()
            model.likelihood.half()

        self.model = model

    def train(self, train_loader, val_loader=None):

        args = self.args
        model = self.model
        likelihood = model.likelihood

        # switch to train mode
        model.train()
        likelihood.train()

        # train
        train_losses = []
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                weight_decay=args.weight_decay)
        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model.gp_layer, num_data=len(train_loader.dataset)
        )
        
        epochs_iter = tqdm(range(args.epochs), desc=f"Training {model.__class__.__qualname__}")
        minibatch_iter = tqdm(train_loader, desc="Minibatch", leave=False)
        with gpytorch.settings.num_likelihood_samples(args.num_mc_train):
            for epoch in epochs_iter:    
                    running_loss = 0.0
                    for (input, target) in minibatch_iter:
                        
                        # enable cuda
                        input, target = input.to(args.device), target.to(args.device)

                        # compute gradient and do SGD step
                        optimizer.zero_grad(set_to_none=True)
                        output = model(input)
                        loss = -mll(output, target)
                        loss.backward()
                        optimizer.step()

                        # record loss
                        running_loss += loss.item() * input.size(0)

                    epochs_iter.set_postfix(loss=loss.item())
                    train_loss = running_loss / len(train_loader.dataset)
                    train_losses.append(train_loss)

                    # validate
                    if args.validate:
                        best_loss = float('inf')
                        val_loss = self.validate(args, val_loader, model, likelihood)
                        if args.verbose:
                            print(f'Epoch {epoch+1}\t\t', 
                                f'Training Loss: {train_loss}\t\t',
                                f'Validation Loss: {val_loss}')
                        if val_loss < best_loss:
                            best_loss = val_loss
                            filename = f'{model.__class__.__qualname__.lower()}_{args.dnn_name}_{train_loader.dataset.dataset.__name__}.pth'
                            save_path = os.path.join(args.checkpoint_dir, filename)
                            torch.save(model.state_dict(), save_path)

        model.likelihood = likelihood

        self.train_loader = train_loader
        self.model = model
        self.train_losses = train_losses
        return model, train_losses

    @staticmethod
    def validate(args, val_loader, model, likelihood):
        # switch to evaluate mode
        model.eval()
        likelihood.eval()

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, model.gp_layer, num_data=len(val_loader.dataset)
        )

        # validate
        running_loss = 0.0
        with torch.no_grad() and gpytorch.settings.num_likelihood_samples(args.num_mc_train):
            for (input, target) in enumerate(val_loader):

                # enable cuda
                input, target = input.to(args.device), target.to(args.device)

                output = model(input)
                loss = -mll(output, target)

                # record loss
                running_loss += loss.item() * input.size(0)
            val_loss = running_loss / len(val_loader.dataset)
        return val_loss



    def test(self, test_loader, model):

        args = self.args
        likelihood = model.likelihood
        
        # initialize
        means_list = []
        stds_list = []
        input_list = []
        target_list = []

        # switch to evaluate mode
        model.eval()
        likelihood.eval()

        # test
        with torch.no_grad() and gpytorch.settings.num_likelihood_samples(args.num_mc_test):
            for (input, target) in test_loader:
                
                # enable cuda
                input, target = input.to(args.device), target.to(args.device)

                # Compute output
                preds = likelihood(model(input))
                means = preds.mean.squeeze(-2)
                stds = preds.variance.squeeze(-2).sqrt()

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