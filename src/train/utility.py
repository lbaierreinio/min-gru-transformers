import torch

def evaluate(model, dataloader, loss_fn, evaluation_type='Validation'):
  with torch.no_grad():
    model.eval()
    total_loss = 0.
    total_correct = 0.
    for batch in dataloader:
      input = batch['input_ids'].cuda()
      labels = batch['labels'].cuda()
      output = model(input)

      # Total loss
      loss = loss_fn(output, labels)
      total_loss += loss.item()

      # Total Correct
      predictions = torch.argmax(output, dim=1)
      total_correct += (predictions == labels).type(torch.float).sum().item()

    accuracy = round(total_correct / len(dataloader.dataset), 4)
    print(f"{evaluation_type} Loss: {round(total_loss, 4)}")
    print(f"{evaluation_type} Accuracy: {accuracy}")
    return accuracy


def train(model, train_dataloader, val_dataloader, num_epochs, loss_fn, learning_rate, *, early_stopping=False):
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  for epoch in range(0, num_epochs+1):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
      input = batch['input_ids'].cuda()
      labels = batch['labels'].cuda()

      optimizer.zero_grad()
      output = model(input)
      loss = loss_fn(output, labels)
      loss.backward()
      total_loss += loss.item()
      optimizer.step()

    # Naive Early Stopping
    if early_stopping and total_loss < 0.25:
      print(f"Early stopping at epoch {epoch}")
      break

    if epoch % 5 == 0:
      print(f"------------ EPOCH {epoch} ------------ \n")
      print(f"Training Loss: {round(total_loss, 4)}")
      accuracy = evaluate(model, val_dataloader, loss_fn, 'Validation')
      if early_stopping and accuracy > 0.95:
        print(f"Early stopping at epoch {epoch}")
        break
      print('\n\n')