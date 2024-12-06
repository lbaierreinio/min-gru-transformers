import time
import torch
import torch.profiler

def evaluate(model, dataloader, loss_fn, evaluation_type='Validation'):
    with torch.no_grad():
        model.eval()
        total_loss = 0.
        total_correct = 0.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for batch in dataloader:
            input = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = ~batch['attention_mask'].to(device).bool()
            output = model(input, mask=mask)

            # Total loss
            loss = loss_fn(output, labels)
            total_loss += loss.item()

            # Total Correct
            predictions = torch.argmax(output, dim=1)
            total_correct += (predictions ==
                              labels).type(torch.float).sum().item()

        accuracy = round(total_correct / len(dataloader.dataset), 4)
        print(f"{evaluation_type} Loss: {round(total_loss, 2)}")
        print(f"{evaluation_type} Accuracy: {accuracy}")
        return total_loss, accuracy


def train_epoch(dataloader, device, model, loss_fn, optimizer):
    training_loss = 0
    total_correct = 0
    epoch_time = 0
    steps = 0
    model.train()
    for batch in dataloader:
        input = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        mask = ~batch['attention_mask'].to(device).bool()
        start = time.time()
        optimizer.zero_grad()
        output = model(input, mask=mask)
        loss = loss_fn(output, labels)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        steps += 1
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time += (time.time() - start)

        predictions = torch.argmax(output, dim=1)
        total_correct += (predictions ==
                        labels).type(torch.float).sum().item()

    return (total_correct / len(dataloader.dataset)), training_loss, total_correct, epoch_time, steps

def train(model, train_dataloader, val_dataloader, num_epochs, loss_fn, optimizer, *, early_stopping_threshold=None, validate_every_i=1, patience=5):
    steps = 0
    total_time = 0
    best_validation_accuracy = 0
    best_validation_loss = float('inf')
    patience_counter = 0
    best_training_accuracy = 0
    best_training_loss = float('inf')
    max_memory = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for epoch in range(0, num_epochs):
        cur_max_memory = 0
        if epoch == 5: # Only profile memory once (significant overhead and does not fluctuate between epochs)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats()    
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
            ) as prof:
                training_accuracy, training_loss, total_correct, epoch_time, epoch_steps = train_epoch(train_dataloader, device, model, loss_fn, optimizer)
                if device.type == 'cuda':
                    cur_max_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                else:
                    for event in prof.key_averages():
                        if event.cpu_memory_usage is not None:
                            cur_max_memory = max(cur_max_memory, event.cpu_memory_usage / (1024 * 1024))
                max_memory = max(max_memory, cur_max_memory)
        else:
            training_accuracy, training_loss, total_correct, epoch_time, epoch_steps = train_epoch(train_dataloader, device, model, loss_fn, optimizer)
        steps += epoch_steps
        # Compute statistics, handle early exiting
        total_time += epoch_time
        best_training_loss = min(best_training_loss, training_loss)
        training_accuracy = total_correct / len(train_dataloader.dataset)
        best_training_accuracy = max(best_training_accuracy, training_accuracy)

        if (epoch+1) % validate_every_i == 0:
            validation_loss, validation_accuracy = evaluate(
                model, val_dataloader, loss_fn, 'Validation')
            
            best_validation_accuracy = max(best_validation_accuracy, validation_accuracy)
            best_validation_loss = min(best_validation_loss, validation_loss)

            print(f"------------ EPOCH {epoch} ------------ \n")
            print(f"Training Loss: {round(training_loss, 4)}")
            print(f"Training Accuracy: {round(training_accuracy, 4)}")    
            print(f"Validation Loss: {round(validation_loss, 4)}")
            print(f"Validation Accuracy: {round(validation_accuracy, 4)}")
            print(f"Epoch Time: {round(epoch_time, 2)}s")
            print(f"Max Memory: {round(cur_max_memory, 2)}MB\n")
            
            results = (round(best_training_loss,2), round(best_validation_loss,2), round(best_training_accuracy,2), round(best_validation_accuracy,2), round(validation_loss,2), round(validation_accuracy,2), steps, epoch+1, round(total_time / (epoch + 1), 2), max_memory)

            if early_stopping_threshold is not None and best_validation_accuracy >= early_stopping_threshold and best_training_accuracy >= early_stopping_threshold:
                print(f"Early stopping at epoch {epoch} due to reaching early stopping threshold")
                return results
            
            if validation_accuracy < (best_training_accuracy - 0.1): # Use 0.1 as model has shown to hover around same validation accuracy on task before starting to learn
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch} due to lack of improvement")
                    return results
            else:
                patience_counter = 0

    validation_loss, validation_accuracy = evaluate(
        model, val_dataloader, loss_fn, 'Validation')
    best_validation_accuracy = max(best_validation_accuracy, validation_accuracy)
    best_validation_loss = min(best_validation_loss, validation_loss)
    return (round(best_training_loss,2), round(best_validation_loss,2), round(best_training_accuracy,2), round(best_validation_accuracy,2), round(validation_loss,2), round(validation_accuracy,2), steps, num_epochs, round(total_time / (epoch + 1), 2), max_memory)
