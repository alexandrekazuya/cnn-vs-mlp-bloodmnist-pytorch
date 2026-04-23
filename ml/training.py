import torch


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.best_epoch = epoch
            return

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return

        self.best_score = score
        self.best_model_state = model.state_dict()
        self.best_epoch = epoch
        self.counter = 0

    def load_best_model(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)


def train_with_validation(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs=100,
    early_stopping=None,
):
    history = {
        "train_loss": [],
        "val_loss": [],
        "stopped_epoch": None,
        "best_epoch": None,
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0

        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_total += loss.item()

        avg_train_loss = train_total / len(train_loader)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.squeeze().long().to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_total += loss.item()

        avg_val_loss = val_total / len(val_loader)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch}\tTraining Loss: {avg_train_loss:.4f}\tValidation Loss: {avg_val_loss:.4f}"
        )

        if early_stopping is not None:
            early_stopping(avg_val_loss, model, epoch)
            if early_stopping.early_stop:
                history["stopped_epoch"] = epoch
                break

    if history["stopped_epoch"] is None:
        history["stopped_epoch"] = epochs

    if early_stopping is not None:
        early_stopping.load_best_model(model)
        history["best_epoch"] = early_stopping.best_epoch

    return history


def evaluate_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total
