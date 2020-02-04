"""
Run this script to train a ConvNet on MNIST.
"""
import torch
from torch import optim, nn
from tqdm import tqdm
import os

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from pytorch_utils.sacred_utils import AuthMongoDbOption
from utils import get_run_dir

from model_ingredient import model_ingredient, make_model
from data_ingredient import data_ingredient, make_dataloaders

torch.backends.cudnn.benchmark = True


ex = Experiment('mnist_classification',
                ingredients=[model_ingredient, data_ingredient])

ex.captured_out_filter = apply_backspaces_and_linefeeds

# ----------------OPTIMIZER-----------------


@ex.config
def optimizer_config():
    """Config for optimzier
    Currently available opts (types of optimizers):
        adam
        adamax
        rmsprop
    """
    lr = 0.001  # learning rate
    opt = 'adam'  # type of optimzier
    weight_decay = 0  # l2 regularization weight_decay (lambda)


@ex.capture
def make_optimizer(model, lr, opt, weight_decay):
    """Make an optimizer of the given type (opt), for the given model's
    parameters with the given learning rate (lr)"""
    optimizers = {
        'adam': optim.Adam,
        'adamax': optim.Adamax,
        'rmsprop': optim.RMSprop,
    }

    optimizer = optimizers[opt](model.parameters(), lr=lr,
                                weight_decay=weight_decay)

    return optimizer


@ex.config
def train_config():
    epochs = 100
    save_every = 1


@ex.automain
def main(epochs, save_every, _log, _run):
    run_dir = get_run_dir(_run)
    os.makedirs(os.path.join(run_dir, "model_statedicts"))

    dset, train_loader, val_loader, test = make_dataloaders()
    model = make_model()
    optimizer = make_optimizer(model)

    lossfn = nn.CrossEntropyLoss()

    for epoch in range(epochs+1):

        # train
        t = tqdm(train_loader)
        t.set_description(desc=f"Epoch: {epoch}")
        for x, y in t:
            y_pred = model(x)
            loss = lossfn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())
        ex.log_scalar("train_loss", loss.item(), epoch)

        # validate
        with torch.no_grad():
            model.eval()
            total_loss = 0
            total_examples = 0
            for x, y in val_loader:
                y_pred = model(x)
                loss = lossfn(y_pred, y)
                total_loss += loss.item()*len(y)
                total_examples += len(y)
            val_loss = total_loss/total_examples
            _log.info(f"val_loss={val_loss:.6f}")
            ex.log_scalar("val_loss", val_loss, epoch)
            model.train()

        # save
        if epoch % save_every == 0:
            name = f"{epoch:03d}model_val{val_loss:.4f}_statedict.pickle"
            path = os.path.join(run_dir, "model_statedicts", name)
            torch.save(model.state_dict(), path)
            ex.add_artifact(path, name)
