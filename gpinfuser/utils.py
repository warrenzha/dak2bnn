import time
import math
import torch

from csv import DictWriter
from copy import deepcopy
from typing import Callable, Iterable, Union, Dict, List
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Progbar:
    def __init__(
            self,
            total: int,
            width: int=30,
            prefix: str='Progress',
            fill: str='.'
        ):

        self.total = total
        self.width = width
        self.prefix = prefix
        self.fill = fill
        self.progress = 0
        self.postfix = None
        
    def set_postfix(self, dict_: dict):
        postfix = ', '.join([f'{key}={value:.4f}' for key, value in dict_.items()])
        self.postfix = ' - ' + postfix     

    def _build_message(self, progress: int) -> str:
        self.progress += progress
        percent = '{0:.1f}'.format(100 * (self.progress / float(self.total)))
        filled = int(self.width * self.progress // self.total)
        bar = self.fill * filled + ' ' * (self.width - filled)
        message = f'{self.prefix} |{bar}| {percent}%'
        return message
        
    def step(self, progress: int=1):
        message = self._build_message(progress)
        if self.postfix is not None:
            message += self.postfix
        print(message, end='\r')

    def close(self):
        print('')

class CSVWriter:
    def __init__(self, filepath: str):
        self.filepath = filepath
        
    def initialize(self, fieldnames: List[str]):
        assert isinstance(fieldnames, list)
        self.csv_file = open(self.filepath, mode='w', newline='')
        self.writer = DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()
        
    def step(self, logs: Dict[str, float]):
        assert isinstance(logs, dict)
        self.writer.writerow(logs)
        
    def close(self):
        self.csv_file.close()

class EnableGradOnPlateau:
    def __init__(
        self,
        params: list[list[torch.nn.Parameter]],
        patience: int=10,
        cooldown: int=0,
        threshold: float=1e-4,
    ):
        params = [list(p) for p in params]
        for p in params:
            require_grad_(p, False)
            
        self.params = params
        self.threshold = threshold
        self.patience = patience
        self.cooldown = cooldown
        self.patience_counter = 0
        self.cooldown_counter = cooldown
        self.best_score = math.inf
        self.last_epoch = 0
        self.total_params = len(params)
        self.current_params = 0
        
    def step(self, score: float):
        if self.has_param:
            score = float(score)
            self.last_epoch += 1

            if self.is_better(score):
                self.best_score = score
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.in_cooldown:
                self.cooldown_counter -= 1
                self.patience_counter = 0

            if self.patience_counter == self.patience:
                require_grad_(self.params[self.current_params], True)
                self.current_params += 1
                self.cooldown_counter = self.cooldown
                self.patience_counter = 0
            
    @property
    def has_param(self):
        return self.current_params < self.total_params
    
    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0
        
    def is_better(self, score: float) -> bool:
        return score < self.best_score - self.threshold
        
    def state_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != 'params'}
    
    def load_state_dict(self, state_dict: dict):
        self.__dict__.update(state_dict)

class MultiStepEnableGrad:
    def __init__(
        self,
        params: list[list[torch.nn.Parameter]],
        milestones: list[int],
    ):
        params = [list(p) for p in params]
        for p in params:
            require_grad_(p, False)
        
        self.params = params
        self.milestones = milestones
        self.last_epoch = 0
        self.current_params = 0
        self.total_params = len(params)
        
    def step(self):
        if self.has_param:
            self.last_epoch += 1
            if self.last_epoch in self.milestones:
                require_grad_(self.params[self.current_params], True)
                self.current_params += 1

    @property
    def has_param(self):
        return self.current_params < self.total_params
        
    def state_dict(self) -> dict:
        return {key: value for key, value in self.__dict__.items() if key != 'params'}
    
    def load_state_dict(self, state_dict: dict):
        self.__dict__.update(state_dict)

class Trainer:
    def __init__(
        self,
        model,
        optimizer: Union[Optimizer, list[Optimizer]],
        scheduler: Union[object, list[object]]=None,
        grad_scheduler: object=None,
        csv_writer: CSVWriter=None,
        metrics: dict[str, Callable]=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_scheduler = grad_scheduler
        self.csv_writer = csv_writer
        self.metrics = metrics

    def zero_grad(self):
        if isinstance(self.optimizer, Iterable):
            for optim in self.optimizer:
                optim.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad(set_to_none=True)
        
    def step(self, batch):
        self.model.train()
        self.zero_grad()
        loss = self.model.compute_loss(batch)
        loss.backward()
        self.step_optimizer()
        return loss.item(), batch[-1].size(0)

    def step_optimizer(self):
        if isinstance(self.optimizer, Iterable):
            for optim in self.optimizer:
                optim.step()
        else:
            self.optimizer.step()

    def step_scheduler(self, score=None):
        if isinstance(self.scheduler, Iterable):
            for sched in self.scheduler:
                if isinstance(sched, ReduceLROnPlateau):
                    sched.step(score)
                else:
                    sched.step()
        else:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(score)
            else:
                self.scheduler.step()
            

    def step_grad_scheduler(self, score=None):
        if isinstance(self.grad_scheduler, EnableGradOnPlateau):
            self.grad_scheduler.step(score)
        else:
            self.grad_scheduler.step()
    
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, scoring: Callable) -> float:
        self.model.eval()

        score, num_data = 0, 0
        for x, y in dataloader:
            batch_size = y.size(0)
            f = self.model.predict(x, num_samples=20)
            score += scoring(y, f) * batch_size
            num_data += batch_size

        return score / num_data
    
    def train(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader=None,
        test_dataloader: DataLoader=None,
        epochs: int=100,
        validation_steps: int=None,
        validation_scoring: Callable=None,
        steps_per_epoch: int=None,
        verbose: bool=True
    ):
        # If provided a evaluation set, then enable early-stopping
        self.do_early_stopping = False if eval_dataloader is None else True
        if self.do_early_stopping:
            assert validation_steps is not None
            assert validation_scoring is not None

        self.validation_curve = []
        self.best_iteration = 0
        self.best_validation_score = math.inf
        validation_steps_count = 0
        stop_training = False

        self.train_time = []
        self.train_curve = []
        if self.metrics is not None:
            self.test_curve = {metric_name: [] for metric_name in self.metrics}
        
        if steps_per_epoch is None:
            steps_per_epoch = len(train_dataloader)

        # Write the results to csv files if a path to a file is provided
        if self.csv_writer is not None:
            fieldnames = ['epoch', 'elapsed_time','train_loss', 'validation_score']
            if self.metrics is not None:
                fieldnames.extend(self.metrics.keys())
            self.csv_writer.initialize(fieldnames)

        # Best parameters based on the `validation_score`
        parameters_state = deepcopy(self.model.state_dict())

        for epoch in range(1, epochs + 1):
            if verbose:
                pbar = Progbar(steps_per_epoch, prefix='Epoch ' + str(epoch))

            # Run epoch
            train_loss, num_data, elapsed_time = 0, 0, time.time()
            for batch in train_dataloader:
                loss, batch_size = self.step(batch)
                train_loss += loss * batch_size
                num_data += batch_size
                
                if verbose:
                    postfix = {'loss': round(train_loss / num_data, 4)}
                    pbar.set_postfix(postfix)
                    pbar.step()
            elapsed_time = time.time() - elapsed_time
            
            # Train loss and validation score
            train_loss /= num_data
            if self.do_early_stopping:
                validation_score = self.evaluate(eval_dataloader, validation_scoring)
            else:
                validation_score = train_loss
            self.train_time.append(elapsed_time)
            self.train_curve.append(train_loss)
            self.validation_curve.append(validation_score)

            if self.csv_writer is not None:
                csv_logs = {
                    'epoch': epoch,
                    'elapsed_time': elapsed_time,
                    'train_loss': train_loss,
                    'validation_score': validation_score
                }

            # If provided, apply the learning rate scheduler
            if self.scheduler is not None:
                self.step_scheduler(validation_score)
 
            # If provided, apply the gradient scheduler
            if self.grad_scheduler is not None:
                self.step_grad_scheduler(validation_score)
            
            # If the `validation_score` improves, save the state of the current parameters
            if validation_score < self.best_validation_score:
                validation_steps_count = 0
                if self.do_early_stopping:
                    parameters_state = deepcopy(self.model.state_dict())
                self.best_iteration = epoch
                self.best_validation_score = validation_score
            else:
                validation_steps_count += 1
                if self.do_early_stopping and (validation_steps_count == validation_steps):
                    stop_training = True
                    
            if verbose:
                postfix['val_score'] = round(self.best_validation_score, 4)
            
            # Compute the test set metrics
            if self.metrics is not None and test_dataloader is not None:
                for metric_name, metric in self.metrics.items():
                    score = self.evaluate(test_dataloader, metric)
                    self.test_curve[metric_name].append(score)
                    
                    if verbose:
                        postfix[metric_name] = round(score, 4)

                    if self.csv_writer is not None:
                        csv_logs[metric_name] = score
            
            if verbose:
                pbar.set_postfix(postfix)
                pbar.step(0)
                pbar.close()

            if self.csv_writer is not None:
                self.csv_writer.step(csv_logs)

            if stop_training:
                break

        if self.csv_writer is not None:
            self.csv_writer.close()

        if self.do_early_stopping:
            self.model.load_state_dict(parameters_state)
        
        return self
    
def require_grad_(params: list[torch.nn.Parameter], flag: bool):
    for p in params:
        p.requires_grad_(flag)