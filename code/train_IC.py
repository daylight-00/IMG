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
import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Dynamically import the config file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def mean_absolute_error(pred, target):
    return torch.mean(torch.abs(pred - target))


def r2_score(pred, target):
    ssr = torch.sum((pred - target) ** 2)
    sst = torch.sum((target - torch.mean(target)) ** 2)
    return 1 - ssr / sst


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    log_file,
    num_epochs,
    patience,
    model_path,
    regularize=False,
    scheduler=None,
):
    model.train()
    best_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - [Training process]: %(message)s",
    )

    for epoch in range(num_epochs):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            break

        start_time = time.time()
        epoch_loss, train_mae, train_r2 = 0, 0, 0
        val_loss, val_mae, val_r2 = 0, 0, 0

        # Training Phase
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            for batch in tepoch:
                # Check if any batch contains None
                if any(x is None for x in batch):
                    print("⚠️ Batch contains None:", batch)
                    continue  # Skip this batch

                tepoch.set_description(f"Epoch {epoch+1}/{num_epochs}")

                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)

                # Forward pass
                outputs = model(*X)
                loss = criterion(outputs, Y)
                if regularize:
                    loss = model.regularize(loss, device)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                # Metrics
                epoch_loss += loss.item() * len(Y)
                train_mae += mean_absolute_error(outputs, Y).item() * len(Y)
                train_r2 += r2_score(outputs, Y).item() * len(Y)

                tepoch.set_postfix(
                    train_loss=loss.item(),
                    mae=train_mae / len(train_loader.dataset),
                    r2=train_r2 / len(train_loader.dataset),
                )


        avg_train_loss = epoch_loss / len(train_loader.dataset)
        avg_train_mae = train_mae / len(train_loader.dataset)
        avg_train_r2 = train_r2 / len(train_loader.dataset)

        # Validation Phase
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                X, Y = batch[:-1], batch[-1]
                X = [x.to(device) for x in X]
                Y = Y.to(device)

                outputs = model(*X)
                loss = criterion(outputs, Y)

                # Metrics
                val_loss += loss.item() * len(Y)
                val_mae += mean_absolute_error(outputs, Y).item() * len(Y)
                val_r2 += r2_score(outputs, Y).item() * len(Y)

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_mae = val_mae / len(val_loader.dataset)
        avg_val_r2 = val_r2 / len(val_loader.dataset)

        epoch_time = int(time.time() - start_time)

        # Logging
        logging.info(
            f"[{model_path}]-[Epoch {epoch+1:03d}/{num_epochs:03d}] - "
            f"Time: {epoch_time} s, "
            f"Train Loss: {avg_train_loss:.5f}, "
            f"Val Loss: {avg_val_loss:.5f}, "
            f"Train MAE: {avg_train_mae:.5f}, "
            f"Val MAE: {avg_val_mae:.5f}, "
            f"Train R²: {avg_train_r2:.5f}, "
            f"Val R²: {avg_val_r2:.5f}"
        )

        print(
            f"[Epoch {epoch+1:03d}/{num_epochs:03d}] "
            f"Time: {epoch_time} s, "
            f"Train Loss: {avg_train_loss:.5f}, "
            f"Val Loss: {avg_val_loss:.5f}, "
            f"Train MAE: {avg_train_mae:.5f}, "
            f"Val MAE: {avg_val_mae:.5f}, "
            f"Train R²: {avg_train_r2:.5f}, "
            f"Val R²: {avg_val_r2:.5f}"
        )

        # Save the best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_path}-best.pt")
            print(f"Best model updated at epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                early_stop = True

        # Save intermediate models
        if (epoch + 1 >= 30) and (epoch + 1 % 5 == 0):
            torch.save(model.state_dict(), f"{model_path}-epoch_{epoch+1}.pt")


def main(config_path):
    # Load configuration
    config = load_config(config_path)
    set_seed(config["seed"])

    # Data preparation
    data_provider = DataProvider(
        epi_path=config["Data"]["epi_path"],
        epi_args=config["Data"]["epi_args"],
        hla_path=config["Data"]["hla_path"],
        hla_args=config["Data"]["hla_args"],
    )
    print(f"Total samples: {len(data_provider)}")

    # Check if any data is None
    for idx in range(len(data_provider)):
        sample = data_provider[idx]
        if any(x is None for x in sample):
            print(f"⚠️ Sample contains None at index {idx}: {sample}")
    
    y = np.array([data_provider.samples[i][2] for i in range(len(data_provider))])
    train_indices, val_indices = train_test_split(
        np.arange(len(data_provider)),
        test_size=config["Data"]["val_size"],
        random_state=42,
    )
    
    full_dataset = config["encoder"](data_provider, **config["encoder_args"])
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["Train"]["batch_size"],
        shuffle=True,
        num_workers=config["Data"]["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["Train"]["batch_size"],
        shuffle=False,
        num_workers=config["Data"]["num_workers"],
    )

    # Model, optimizer, and criterion setup
    model = config["model"](**config["model_args"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = config["Train"]["criterion"]()
    optimizer = config["Train"]["optimizer"](model.parameters(), **config["Train"]["optimizer_args"])

    scheduler = None
    if config["Train"]["use_scheduler"]:
        from utils.scheduler import CosineAnnealingWarmUpRestarts

        total_training_steps = config["Train"]["num_epochs"] * len(train_loader)
        cycle_len = total_training_steps // 10
        warmup_steps = cycle_len // 10
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=cycle_len,
            T_mult=1,
            T_up=warmup_steps,
            eta_max=0.1,
            gamma=0.8,
        )

    # Start training
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        log_file=config["log_file"],
        num_epochs=config["Train"]["num_epochs"],
        patience=config["Train"]["patience"],
        model_path=os.path.join(config["chkp_path"], config["chkp_name"]),
        regularize=config["Train"]["regularize"],
        scheduler=scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model with specified config.")
    parser.add_argument("config_path", type=str, help="Path to the config.py file.")
    args = parser.parse_args()
    main(args.config_path)
