import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from evaluate import evaluate


def trainer(training_params):
    """
    Train the model and return a dictionary.

    Args:
        model: The model to be trained.
        train_loader: DataLoader for the training dataset.
        valid_loader: DataLoader for the validation dataset.
        device: Device to run the training on ('cuda' or 'cpu').
        training_params: Dictionary containing 'optimizer',
        'scheduler', and optionally 'scaler'.

        num_epochs: Number of training epochs.

    Returns:
        dict: Contains the trained model, optimizer, scheduler,
        scaler, best mAP, best epoch, and history.
    """

    model = training_params.get('model')
    train_loader = training_params.get('train_loader')
    valid_loader = training_params.get('valid_loader')

    optimizer = training_params.get("optimizer")
    scheduler = training_params.get("scheduler")

    device = training_params.get("device")
    num_epochs = training_params.get('num_epochs')
    save_path = training_params.get('save_path')

    scaler = training_params.get("scaler", GradScaler())

    history = {
        "epoch_loss": [],
        "mAP": []
    }
    best_mAP = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for imgs, tars in bar:
            images = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in tars]

            optimizer.zero_grad()
            with autocast(device_type='cuda', enabled=True):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += losses.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} Training Loss: {epoch_loss:.4f}")
        history["epoch_loss"].append(epoch_loss)

        print("Evaluating on validation set...")
        mean_ap = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch+1} mAP: {mean_ap:.4f}")
        history["mAP"].append(mean_ap)

        if best_mAP < mean_ap:
            best_mAP = mean_ap
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path + "_best.pth")

    torch.save(model.state_dict(), save_path + "_last.pth")
    print("Training complete and model saved!")

    results = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "best_mAP": best_mAP,
        "best_epoch": best_epoch,
        "history": history
    }

    return results
