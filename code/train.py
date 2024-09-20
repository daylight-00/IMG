# train.py

import sys, os
sys.path.insert(0, os.getcwd())
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
import logging
import time
import importlib.util
import argparse
from dataprovider import DataProvider

def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config

def train_model(model, train_loader, val_loader, criterion, optimizer, device, log_file, num_epochs, patience, model_path, regularize=False):
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - [Training process]: %(message)s'
    )
    for epoch in range(num_epochs):
        if early_stop:
            print(f'Early stopping at epoch {epoch}')
            break

        start_time = time.time()
        epoch_loss = 0
        val_loss = 0
        correct_train = 0
        total_train = 0
        correct_val = 0
        total_val = 0

        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                outputs = model(*X)
                loss = criterion(outputs, Y)
                if regularize:
                    loss = model.regularize(loss, device)
                    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())
                epoch_loss += loss.item()

                # Accuracy 계산
                predicted = (outputs > 0.5).float()
                correct_train += (predicted == Y).sum().item()
                total_train += Y.size(0)

        avg_train_loss = epoch_loss / len(train_loader)
        train_acc = correct_train / total_train

        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)
                
                outputs = model(*X)
                loss = criterion(outputs, Y)
                val_loss += loss.item()

                # Accuracy 계산
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == Y).sum().item()
                total_val += Y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_val / total_val

        epoch_time = int(time.time() - start_time)
        logging.info(f'\
[{model_path}]-[Epoch {epoch+1:03d}/{num_epochs:03d}] - \
Time: {epoch_time} s, \
Train Acc: {train_acc:.5f}, \
Val Acc: {val_acc:.5f}, \
Train Loss: {avg_train_loss:.5f}, \
Val Loss: {avg_val_loss:.5f}'\
)
        torch.save(model.state_dict(), f"{model_path}-epoch_{epoch+1}.pt")
        
        # Early stopping and best model saving
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), f"{model_path}-best.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

def main(config_path):
    # Load configuration
    config = load_config(config_path)

    # Create DataProvider instance
    data_provider = DataProvider(
        epi_path=config["Data"]["epi_path"],
        epi_args=config["Data"]["epi_args"],
        hla_path=config["Data"]["hla_path"],
        hla_args=config["Data"]["hla_args"],
    )
    print(f"Total samples: {len(data_provider)}")

    # Extract targets for stratified splitting
    y = np.array([data_provider.samples[i][2] for i in range(len(data_provider))])

    # Split indices into training and validation sets
    train_indices, val_indices = train_test_split(
        np.arange(len(data_provider)),
        stratify=y,
        test_size=config["Data"]["val_size"],
        random_state=42
    )

    # Create EncodedDataset instances
    full_dataset = config["encoder"](data_provider, **config["encoder_args"])

    # Create subsets for training and validation
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    print(f"Samples in train set: {len(train_dataset)}")
    print(f"Samples in validation set: {len(val_dataset)}")

    # Create DataLoaders with multiple workers for parallel data loading
    batch_size = config["Train"]["batch_size"]
    num_workers = config["Data"]["num_workers"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Initialize model, criterion, optimizer, etc.
    model = config["model"](**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f'Model loaded on {device}')

    criterion = config["Train"]["criterion"]()
    optimizer = config["Train"]["optimizer"](
        model.parameters(),
        **config["Train"]["optimizer_args"]
    )

    # Start training
    train_model(
        model           = model,
        train_loader    = train_loader,
        val_loader      = val_loader,
        criterion       = criterion,
        optimizer       = optimizer,
        device          = device,
        log_file        = config["log_file"],
        num_epochs      = config["Train"]["num_epochs"],
        patience        = config["Train"]["patience"],
        regularize      = config["Train"]["regularize"],
        model_path      = os.path.join(config["chkp_path"], config["chkp_name"])
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()
    main(args.config_path)