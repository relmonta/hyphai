import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from .dataset import CloudDataset, Rotate, ToTensor
from .models import *
from .utils import *
from .metrics import *
import time
import pickle
import torch
import os

# Path where to save the models and the metrics
SCRATCH = #TODO
# Path to the repository
REPO_PATH = #TODO
# Path to the data
DATA_PATH = #TODO

class Trainer:
    """Trainer class.

    Args:
        model (torch.nn.Module) : Model to train.
        train_loader (torch.utils.data.DataLoader) : Training data.
        optimizer (torch.optim.Optimizer) : Optimizer.
        criterion (torch.nn.Module) : Loss function.
        save_step (int) : Number of steps between each snapshot save.
        path_name (str) : Path to save snapshots.
        metrics_path (str) : Path to save metrics.
        clip_value (float) : Clip value for gradient clipping.
        resume (bool) : Whether to resume training or not.
        val_loader (torch.utils.data.DataLoader) : Validation data.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: torch.nn.Module,
        save_step: int,
        path_name: str,
        metrics_path: str,
        clip_value: float = 0.0,
        resume: bool = False,
        val_loader: DataLoader = None,
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print("Using GPU {}".format(self.device))
        self.model = model
        self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_val_samples = 256
        self.metrics_path = os.path.join(SCRATCH, metrics_path)
        if not os.path.exists(self.metrics_path):
            os.makedirs(self.metrics_path)
        self.optimizer = optimizer
        self.criterion = criterion
        self.regularization = torch.nn.MSELoss()
        self.len_train_loader = len(train_loader)
        self.clip_value = clip_value
        self.leadtime = self.train_loader.dataset.leadtime
        self.n_classes = self.train_loader.dataset.n_classes
        self.max_f1_score = 0
        self.losses = []
        self.accs = []
        self.save_step = save_step
        self.epochs_done = 0
        self.snapshot_path = f'{SCRATCH}/snapshots/{path_name}'
        self.weights_path = f'{SCRATCH}/models/{path_name}'
        # Set gpu for the model
        self.model.set_device(self.device)
        if resume and os.path.exists(self.snapshot_path):
            self._load_snapshot(self.snapshot_path)

    def _load_snapshot(self, snapshot_path):
        """Loads a snapshot of the model.

        Args:
            snapshot_path (str) : Path to the snapshot.
        """
        import datetime
        print("Loading snapshot from {}".format(snapshot_path))
        print('Modified on:', datetime.datetime.fromtimestamp(
            os.path.getmtime(snapshot_path)))
        # Load the snapshot
        snapshot = torch.load(
            snapshot_path, map_location=self.device)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_done = snapshot["EPOCHS_DONE"]
        if "MAX_VAL_F1_SCORE" in snapshot:
            self.max_f1_score = snapshot["MAX_VAL_F1_SCORE"]
        else:
            self.max_f1_score = 0
        print(f"Model loaded from snapshot at Epoch {self.epochs_done}")

    def _to_cuda(self, x):
        """Moves a tensor or a list of tensors to the GPU.

        Args:
            x (Tensor or list of Tensors) : Tensor or list of tensors to move to the GPU.

        Returns:
            Tensor or list of Tensors : Tensor or list of tensors on the GPU.
        """
        if torch.is_tensor(x):
            return x.to(self.device)
        elif isinstance(x, list):
            x_cuda = []
            for element in x:
                x_cuda.append(self._to_cuda(element))
            return x_cuda
        else:
            raise TypeError("Invalid type for to_cuda")

    def _progress_bar(self, i):
        """Prints a progress bar with the current progress."""
        return 25 * "=" if i == self.len_train_loader - 1 else "=" * (25 * i // self.len_train_loader + 1) + ">" + "." * \
            (25 * (self.len_train_loader - i) // self.len_train_loader - 1)

    def _print_logs(self, epoch, i):
        """Prints the results of the current epoch and iteration.

        Args:
            epoch (int) : Current epoch.
            i (int) : Current iteration.
        """

        logs = f'Epoch {epoch+1}/{self.max_epochs} {i + 1:3d}/{self.len_train_loader} ' + \
            f'[{self._progress_bar(i)}] ET: {int(time.time()-self.start_time):.1f}s' + \
            f' < {int(time.time()-self.start_time)*(self.len_train_loader-i-1)/(i+1):.1f}s' + \
            f' loss: {self.train_loss / (i+1):.5f} |' + \
            f' f1_score_{self.leadtime*15}min: {100*self.train_f1_score/(i+1):.3f}% |' + \
            f' f1_score_15min: {100*self.train_f1_score_15/(i+1):.3f}% |' + \
            f' acc_{self.leadtime*15}min: {100*self.train_acc/(i+1):.2f}% |' + \
            f' acc_15min: {100*self.train_acc_15/(i+1):.2f}% |'
        print(logs, end='\r' if i <
              self.len_train_loader - 1 else '\n')

        # Write logs to file
        with open(REPO_PATH + "src/logs.txt", "a") as f:
           f.write(logs + '\n')

    def _save_snapshot(self, epoch):
        """Saves the current model if the accuracy is higher than the previous one."""
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.state_dict()
        snapshot["EPOCHS_DONE"] = epoch
        snapshot["MAX_F1_SCORE"] = self.train_f1_score
        snapshot["MAX_VAL_F1_SCORE"] = self.val_f1_score
        snapshot["MAX_VAL_F1_SCORE_15"] = self.val_f1_score_15
        # Save model if accuracy is higher than previous one
        if self.val_f1_score > self.max_f1_score:
            torch.save(snapshot, self.weights_path)
            print(
                f'Model saved with F1 score = {self.val_f1_score:.3f}%')
            # Update max accuracy
            self.max_f1_score = self.val_f1_score
        if epoch % self.save_step == 0:
            torch.save(snapshot, self.snapshot_path)
            print(f'Snapshot saved at Epoch {epoch}')

    def _run_batch(self, inputs, targets, update_weights=True):
        """Runs a batch of training.

        Args:
            inputs (torch.Tensor) : Input batch.
            targets (torch.Tensor) : Target batch.
            update_weights (bool) : Whether to update weights or not.
        """
        # forward pass
        outputs = self.model(inputs)
        # loss over the first time step
        loss = self.criterion(outputs[0], targets[0])
        # loss over the remaining time steps
        for t in range(1, len(outputs)):
            loss += self.criterion(outputs[t], targets[t])
        # Check if targets are one-hot encoded
        if targets[0].shape[1] == self.n_classes:
            # targets[i] (B, n_classes, H, W) -> (B, H, W)
            labels = [targets[t].argmax(dim=1) for t in range(len(targets))]
        else:
            labels = targets
        # Check if loss is not nan and does not explode. If it does, do not compute the gradients.
        if not np.isnan(loss.item()) and loss.item() < 100: 
            loss /= self.accumulations_steps
            # print statistics
            self.train_loss += float(loss)
            # Backward pass
            loss.backward()
            # Clip gradients
            if self.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_value)
            # Compute metrics
            outputs = [outputs[t].unsqueeze(0)
                       for t in range(len(outputs))]
            outputs = torch.cat(outputs)
            # Get labels
            pred = np.argmax(outputs.cpu().detach().numpy(), axis=-3)
            y_true = [labels[t].unsqueeze(0) for t in range(self.leadtime)]
            y_true = torch.cat(y_true).cpu().numpy()
            obs = y_true[:, :, MRG:-MRG, MRG:-MRG]
            pred = pred[:, :, MRG:-MRG, MRG:-MRG]
            batch_size = pred.shape[1]
            f1_tmp = np.zeros((batch_size, self.leadtime))
            for j in range(batch_size):
                # macro-averaged F1 score
                f1_tmp[j], _, _ = avg_f1_score(
                    obs[:, j, ...], pred[:, j, ...], average='macro')
            # Get the training F1 score
            self.train_f1_score += np.mean(f1_tmp[:, -1])
            self.train_f1_score_15 += np.mean(f1_tmp[:, 0])
            self.train_acc += np.mean(np.equal(obs[-1], pred[-1]))
            self.train_acc_15 += np.mean(np.equal(obs[0], pred[0]))
            # Update weights every self.accumulations_steps
            # This is equivalent to having a batch size of
            # self.accumulations_steps*batch_size
            if update_weights:
                # update weights
                self.optimizer.step()
                # The tensors contains the gradients of the previous accumulation steps
                # so we need to zero the parameter gradients after the weights
                # are updated
                self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad() # Disable gradient tracking
    def _run_validation(self, epoch):
        """Runs a validation epoch.

        Args:
            epoch (int) : Current epoch.
        """

        # Iterate over validation data but stop after self.n_val_samples
        # Reset loss and accuracy
        metrics_val_dict = None  # Dictionary to store the metrics : "avg_cce", "avg_acc",
        # "avg_iou", "avg_ma_f1", "avg_precision", "avg_recall", "confusion_matrix", "avg_hausdorff"
        for i, data in enumerate(self.val_loader):
            # Pass data to GPU
            if isinstance(self.model, FullDLModel):
                inputs = self._to_cuda(data['X'][0])
            else:
                inputs = self._to_cuda(data['X'])
            # check if inputs are free of nan
            if isnan(inputs):
                print("NaN in inputs")
                continue
            # Check if targets are one-hot encoded
            if data['y'][0].shape[1] == self.n_classes:
                # targets[i] (B, n_classes, H, W) -> (B, H, W)
                labels = [data['y'][t].argmax(dim=1) for t in range(len(data['y']))]
            else:
                labels = data['y']

            # Forward pass
            with torch.no_grad():
                outputs = self.model(inputs)
            outputs = [outputs[t].unsqueeze(0)
                        for t in range(len(outputs))]
            outputs = torch.cat(outputs)
            # Get labels
            pred = np.argmax(outputs.cpu().detach().numpy(), axis=-3)
            # Implemented metrics expect the probabilities to be in the last axis
            prob = np.moveaxis(outputs.cpu().detach().numpy(), -3, -1)
            # Compute metrics
            y_true = [labels[t].unsqueeze(0)
                        for t in range(len(labels))]
            y_true = torch.cat(y_true).cpu().numpy()
            batch_size = pred.shape[1]
            for j in range(batch_size):
                metrics_val_dict = compute_val_metrics(pred[:, j, ...], prob[:,
                                                                                j, ...], y_true[:, j, ...], self.leadtime,
                                                        self.n_classes, metrics_val_dict, self.n_val_samples,
                                                        add_hausdorff=False)
            if metrics_val_dict['steps_done'] >= self.n_val_samples:
                break
        self.val_f1_score = np.mean(metrics_val_dict["avg_ma_f1"], axis=0)[-1]
        self.val_f1_score_15 = np.mean(metrics_val_dict["avg_ma_f1"], axis=0)[0]
        # Print logs
        logs = f'Validation Epoch {epoch} - '
        logs += f'avg Acc (15min): {np.mean(metrics_val_dict["avg_acc"], axis=0)[0]*100:.2f}% - '
        logs += f'avg Acc (120min): {np.mean(metrics_val_dict["avg_acc"], axis=0)[-1]*100:.2f}% - '
        logs += f'avg F1 (15min): {np.mean(metrics_val_dict["avg_ma_f1"], axis=0)[0]*100:.2f}% - '
        logs += f'avg F1 (120min): {np.mean(metrics_val_dict["avg_ma_f1"], axis=0)[-1]*100:.2f}% - '
        print(logs)
        # Save metrics to already existing pkl file containing scores from previous epochs and for this model
        val_metrics_path = os.path.join(self.metrics_path, 'val_metrics.pkl')
        if os.path.exists(val_metrics_path):
            with open(val_metrics_path, 'rb') as f:
                val_metrics = pickle.load(f)
        else:
            val_metrics = {}
        val_metrics[epoch] = metrics_val_dict
        with open(val_metrics_path, 'wb') as f:
            pickle.dump(val_metrics, f, pickle.HIGHEST_PROTOCOL)
        # Iterate over training data but stop after self.n_val_samples
        metrics_train_dict = None
        for i, data in enumerate(self.train_loader):
            # Pass data to GPU
            if isinstance(self.model, FullDLModel):
                inputs = self._to_cuda(data['X'][0])
            else:
                inputs = self._to_cuda(data['X'])
            # Check if targets are one-hot encoded
            if data['y'][0].shape[1] == self.n_classes:
                # targets[i] (B, n_classes, H, W) -> (B, H, W)
                labels = [data['y'][t].argmax(dim=1) for t in range(len(data['y']))]
            else:
                labels = data['y']                    
            # Forward pass
            with torch.no_grad():
                outputs = self.model(inputs)
            outputs = [outputs[t].unsqueeze(0)
                        for t in range(len(outputs))]
            outputs = torch.cat(outputs)
            # Get labels
            pred = np.argmax(outputs.cpu().detach().numpy(), axis=-3)
            # Implemented metrics expect the probabilities to be in the last axis
            prob = np.moveaxis(outputs.cpu().detach().numpy(), -3, -1)
            # Compute metrics
            y_true = [labels[t].unsqueeze(0)
                        for t in range(len(labels))]
            y_true = torch.cat(y_true).cpu().numpy()
            batch_size = pred.shape[1]
            for j in range(batch_size):
                metrics_train_dict = compute_val_metrics(pred[:, j, ...], prob[:,
                                                                                j, ...], y_true[:, j, ...], self.leadtime,
                                                            self.n_classes, metrics_train_dict, self.n_val_samples,
                                                            add_hausdorff=False)
            if metrics_train_dict['steps_done'] >= self.n_val_samples:
                break
        # Save metrics to already existing pkl file containing scores from previous epochs and for this model
        train_metrics_path = os.path.join(
            self.metrics_path, 'train_metrics.pkl')
        if os.path.exists(train_metrics_path):
            with open(train_metrics_path, 'rb') as f:
                train_metrics = pickle.load(f)
        else:
            train_metrics = {}
        train_metrics[epoch] = metrics_train_dict
        with open(train_metrics_path, 'wb') as f:
            pickle.dump(train_metrics, f, pickle.HIGHEST_PROTOCOL)

    def _run_epoch(self, epoch):
        """Runs an epoch of training.

        Args:
            epoch (int) : Current epoch.
        """
        # Set start time
        self.start_time = time.time()
        self.train_loss = 0
        self.train_acc = 0
        self.train_acc_15 = 0
        self.train_f1_score = 0
        self.train_f1_score_15 = 0
        # Iterate over training data
        for i, data in enumerate(self.train_loader):
            # Pass data to GPU, _to_cuda is a function that can handle both
            # single tensor and list of tensors
            # The targets data['y'] are a list of tensors, each tensor
            # corresponding to a time step
            if isinstance(self.model, FullDLModel):
                inputs = self._to_cuda(data['X'][0])
            else:
                inputs = self._to_cuda(data['X'])
            targets = self._to_cuda(data['y'])
            # Run over the batch
            update_weights = (i+1) % self.accumulations_steps == 0
            self._run_batch(
                inputs, targets, update_weights=update_weights)
            # Compute metrics
            if update_weights:
                # Print logs
                self._print_logs(epoch, i)
        self.train_loss /= self.len_train_loader
        self.train_acc /= self.len_train_loader
        self.train_acc_15 /= self.len_train_loader 
        self.train_f1_score /= self.len_train_loader
        self.train_f1_score_15 /= self.len_train_loader


    def train(self, max_epochs: int, accumulations_steps: int = 1):
        """Trains the model.

        Args:
            max_epochs (int) : Number of epochs to train the model.
        """
        print(
            f'Training model for {max_epochs} epochs on GPU {self.device}, with '
            + f'{self.train_loader.dataset.context_size} input obs {self.train_loader.dataset.leadtime} lead times')
        print(
            f"Training set size: {self.len_train_loader} batches of size {self.train_loader.batch_size}")
        self.max_epochs = max_epochs
        # Set model to train mode
        self.model.train()
        # Number of steps before updating weights, equivalent to having a batch size of accumulations_steps * batch_size
        self.accumulations_steps = accumulations_steps
        for epoch in range(self.epochs_done, max_epochs):
            self._run_epoch(epoch)

            # validation step
            self.model.eval()
            self._run_validation(epoch)
            self.model.train()

            # The checkpoint is saved only by the first process
            self._save_snapshot(epoch+1)

        print('Finished Training')

def prepare_dataloader(phase: str, batch_size: int, context_size: int, leadtime: int, n_classes: int, target_one_hot: bool = False, proportion: float = 1.0):
    """Prepares the dataloader.

    Args:
        phase (str) : Phase of the model ('train', 'val', 'test').
        batch_size (int) : Batch size.
        context_size (int) : Number of frames used as context.
        leadtime (int) : Max lead time.
        n_classes (int) : Number of classes.
        target_one_hot (bool) : If True, the targets are one-hot encoded.
        proportion (float) : Proportion of the dataset to use.

    Returns:
        torch.utils.data.DataLoader : Dataloader.
    """
    # 1st case : 3 classes categories, 0: no cloud, 1: low clouds, 2: high clouds
    # each category contains several classes
    # LEVELS = {}
    # LEVELS[0] = [0, 6]
    # LEVELS[1] = [1, 2, 3]
    # LEVELS[2] = [4, 5, 7, 8, 9, 10, 11]

    # 2nd case : all the classes are considered each one as a class
    LEVELS = {}
    for i in range(12):
        LEVELS[i] = [i]
    n_classes = len(LEVELS.keys())

    # Load the partition dictionary (train, validation, test)
    # with open(f'{REPO_PATH}src/saved_partition.pkl', 'rb') as f:
    #     tmp = pickle.load(f)
    partition = {}
    # 100 sequences of 12 images are provided in zenodo. To obtain the entire dataset, please contact the data provider, EUMETSAT.
    partition['train'] = np.arange(1, 65)
    partition['validation'] = np.arange(64, 101)
    # Set proportion of the training set
    set_length = int(partition['train'].shape[0]*proportion)
    partition['train'] = partition['train'][:set_length]

    params = {'n_classes': n_classes,
              # number of observations to consider as context (t, t-1,..., t-context_size + 1)
              'context_size': context_size,
              'leadtime': leadtime,  # max lead time
              'levels': LEVELS}
    # Augment the training set
    if phase == 'train':
        transform = transforms.Compose([
            Rotate(),
            ToTensor(target_one_hot=target_one_hot)])
    # Validation set is not augmented
    else:
        transform = transforms.Compose(
            [ToTensor(target_one_hot=target_one_hot)])
        
    dataset = CloudDataset(DATA_PATH, partition[phase], **params, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,  # CUDA only
        # Don't shuffle for distributed training (shuffle is done by the
        # sampler)
        shuffle=phase == 'train',
        num_workers=1,
    )
    return loader

def main():
    torch.cuda.empty_cache()
    model_name = "hyphai-1"
    total_epochs = 30
    batch_size = 4
    learning_rate = 1e-3
    accumulations_steps = 16
    context_size = 4
    maxleadtime = 8
    # Number of classes
    N_CLASSES = 12
    freeze_unet_velocity = False
    # Resume training from the last checkpoint
    resume = False
    # Initialize the model
    if model_name.lower() == 'full-dl':
        model = FullDLModel(n_classes=N_CLASSES,
                             context_size=context_size, leadtime=maxleadtime)
        # Paths where to save model weights, checkpoints and training + validation metrics
        path = f'full-dl'
        model_path = f'{path}.pt'
        metrics_path = f'metrics/{path}'
        print("Using full-dl model")
    elif model_name.lower() == 'hyphai-2':
        model = HyPhAI2(n_classes=N_CLASSES, context_size=context_size, leadtime=maxleadtime)
        pretrained_dict = torch.load("hyphai_1.pt")["MODEL_STATE"]
        # Remove the "unet." prefix from the keys 
        pretrained_dict = {key.replace("unet.", ""): value for key, value in pretrained_dict.items()}
        model.unet.load_state_dict(pretrained_dict)
        # Freeze the weights of the unet used for the velocity field
        # 
        if freeze_unet_velocity:
            for p in model.unet.parameters():
                p.requires_grad = False    
        path = f'hyphai_2'
        model_path = f'{path}.pt'
        metrics_path = f'metrics/{path}'
        print(f"Using HyPhAI-2 model") 
    else:
        model = HyPhAI1(n_classes=N_CLASSES,context_size=context_size, leadtime=maxleadtime)
        path = f'hyphai_1'
        model_path = f'{path}.pt'
        metrics_path = f'metrics/{path}'
        print("Using hyphai-1 model")      
    # Initialize the optimizer and the loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    # Prepare the dataloader
    train_loader = prepare_dataloader(
        'train', batch_size, context_size, maxleadtime, N_CLASSES, target_one_hot=isinstance(criterion, torch.nn.MSELoss))
    val_loader = prepare_dataloader(
        'validation', batch_size, context_size, maxleadtime, N_CLASSES, target_one_hot=isinstance(criterion, torch.nn.MSELoss))
    # Initialize the trainer
    trainer = Trainer(model, train_loader, optimizer, criterion, val_loader=val_loader, save_step=1, path_name=model_path, metrics_path=metrics_path, resume=resume)
    # Train the model
    trainer.train(total_epochs, accumulations_steps=accumulations_steps)
