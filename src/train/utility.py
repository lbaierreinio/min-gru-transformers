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
            mask = ~batch['attention_mask'].to(device).bool()
            output = model(input, mask)

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


def train(model, train_dataloader, val_dataloader, num_epochs, loss_fn, learning_rate, *, early_stopping_threshold=None, check_every_i=1, accumulate_every_i=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    steps = 0
    total_time = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0.0


        data_iterator = iter(train_dataloader)
        idx = 0
        while idx < len(train_dataloader): # Iterate over minibatches
            max_idx = min(idx + accumulate_every_i, len(train_dataloader))
            batches = [next(data_iterator) for _ in range(idx, max_idx)] # Retrieve set of minibatches
            num_examples = sum([len(batch['input_ids']) for batch in batches])
            for batch in batches: # Iterate over each minibatch
                input = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                mask = ~batch['attention_mask'].to(device).bool()

                t0 = time.time()
                # Backward & forward pass
                output = model(input, mask=mask)
                loss = loss_fn(output, labels) * (len(batch['input_ids']) / num_examples) # Scale loss by number of examples in minibatch
                total_loss += loss.item()
                loss.backward()
                            
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                total_time += (time.time() - t0)

            idx = max_idx
            
            # Step with accumulated gradients
            optimizer.step()
            optimizer.zero_grad()
            steps += 1

        if (epoch + 1) % check_every_i == 0:
            print(f"------------ EPOCH {epoch} ------------ \n")
            print(f"Training Loss: {round(total_loss, 4)}")
            total_loss, accuracy = evaluate(
                model, val_dataloader, loss_fn, 'Validation')
            if early_stopping_threshold is not None and accuracy >= early_stopping_threshold:
                print(f"Early stopping at epoch {epoch}")
                return total_loss, accuracy, steps, epoch, (total_time / epoch)
            print('\n\n')

    total_loss, accuracy = evaluate(
        model, val_dataloader, loss_fn, 'Validation')
    return total_loss, accuracy, steps, num_epochs, (total_time / num_epochs)
