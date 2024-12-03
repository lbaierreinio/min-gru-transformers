import time
import torch


def evaluate(model, dataloader, loss_fn, evaluation_type='Validation'):
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        total_correct = 0.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for batch in dataloader:
            input = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            output = model(input)

            # Total loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()

            # Total Correct
            predictions = torch.argmax(output, dim=1)
            total_correct += (predictions ==
                              labels).type(torch.float).sum().item()

        accuracy = round(total_correct / len(dataloader.dataset), 4)
        print(f"{evaluation_type} Loss: {round(total_loss, 4)}")
        print(f"{evaluation_type} Accuracy: {accuracy}")
        return total_loss, accuracy


def train(model, train_dataloader, val_dataloader, num_epochs, loss_fn, learning_rate, *, early_stopping_threshold=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = 0
    total_time = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            input = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = ~batch['attention_mask'].to(device).bool()

            start = time.time()  # TODO: Verify this is a good way to measure time b/c batch size might not necessarily always be the same(?)
            optimizer.zero_grad()
            output = model(input, mask=mask)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)
            steps += 1

        # Naive Early Stopping (TODO: Revisit if this is a good idea & also consider checking every Epoch for improved accuracy)
        if epoch % 5 == 0:
            print(f"------------ EPOCH {epoch} ------------ \n")
            print(f"Training Loss: {round(total_loss, 4)}")
            total_loss, accuracy = evaluate(
                model, val_dataloader, loss_fn, 'Validation')
            if early_stopping_threshold is not None and accuracy >= early_stopping_threshold:
                print(f"Early stopping at epoch {epoch}")
                return total_loss, accuracy, steps, epoch, (total_time / steps)
            print('\n\n')

    total_loss, accuracy = evaluate(
        model, val_dataloader, loss_fn, 'Validation')
    return total_loss, accuracy, steps, num_epochs, (total_time / steps)
