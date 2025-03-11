import json
import torch
import transformers
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

from DecoderTrans import TransformerDecoder
from tokenizer import CurriculumTokenizer
import config


with open('config.json', 'r') as file:
    data = json.load(file)

config = config.Config(**data)

class ExpressionDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.tokenizer = CurriculumTokenizer()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        expression = row['expression']
        result = row['result']
        result -= self.df['result'].min()
        encoded_expression = self.tokenizer.encode(expression)
        return encoded_expression, result


def collate_batch(batch):
    expressions, results = zip(*batch)
    expressions = [torch.tensor(expr) for expr in expressions]
    lengths = [len(expr) for expr in expressions]
    padded_expressions = pad_sequence(expressions, batch_first=True, padding_value=0)
    results_tensor = torch.LongTensor(results)
    return padded_expressions, lengths, results_tensor


def train_epoch(model, loader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 3:
            inputs, _, targets = batch
        elif len(batch) == 2:
            inputs, targets = batch
        else:
            raise ValueError("Unexpected batch format.")

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        targets_expanded = targets.unsqueeze(1).expand(-1, outputs.size(1))

        outputs = outputs.view(-1, outputs.size(-1))  # (32 * 17, 17746)
        targets_expanded = targets_expanded.reshape(-1)

        loss = criterion(outputs, targets_expanded)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == targets_expanded).sum().item()/inputs.size(1)
        total += targets.size(0)

        writer.add_scalar("Train/Loss", loss.item(), epoch * len(loader) + batch_idx)

    scheduler.step()

    accuracy = 100 * correct / total
    epoch_loss = total_loss / len(loader)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)
    writer.add_scalar("Train/EpochLoss", epoch_loss, epoch)
    return epoch_loss, accuracy


def evaluate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                inputs, _, targets = batch
            elif len(batch) == 2:
                inputs, targets = batch
            else:
                raise ValueError("Unexpected batch format.")

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            targets_expanded = targets.unsqueeze(1).expand(-1, outputs.size(1))

            outputs = outputs.view(-1, outputs.size(-1))  # (32 * 17, 17746)
            targets_expanded = targets_expanded.reshape(-1)

            loss = criterion(outputs, targets_expanded)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets_expanded).sum().item()/inputs.size(1)
            total += targets.size(0)

            writer.add_scalar("Val/Loss", loss.item(), epoch * len(loader) + batch_idx)

    accuracy = 100 * correct / total
    epoch_loss = total_loss / len(loader)
    writer.add_scalar("Val/Accuracy", accuracy, epoch)
    writer.add_scalar("Val/EpochLoss", epoch_loss, epoch)
    return epoch_loss, accuracy

def log_final_plots(epochs, random_losses, curriculum_losses, random_accs, curriculum_accs):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].plot(range(1, epochs + 1), random_losses, label='Random Loss')
    ax[0].plot(range(1, epochs + 1), curriculum_losses, label='Curriculum Loss')
    ax[0].set_title('Loss Comparison')
    ax[0].legend()

    ax[1].plot(range(1, epochs + 1), random_accs, label='Random Accuracy')
    ax[1].plot(range(1, epochs + 1), curriculum_accs, label='Curriculum Accuracy')
    ax[1].set_title('Accuracy Comparison')
    ax[1].legend()

    writer.add_figure("Results/Loss_Accuracy", fig)
    plt.close(fig)


# Случайное сэмплирование
def train_learning(model, train_loader, val_loader, optimizer, criterion, scheduler, device, epochs):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(epochs):
        print(f"\nRandom Sampling Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, scheduler, device, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    return train_losses, val_losses, train_accs, val_accs


writer = SummaryWriter(config.logging_dir)

device = torch.device(config.training.device)

df = pd.read_csv(config.data.dataset_path)

def clean_expression(expression):
    expression = expression.replace(" ", "")
    return expression

df['expression'] = df['expression'].apply(clean_expression)

train_df, val_df = train_test_split(df, test_size=config.data.train_split, random_state=config.data.random_seed)
train_dataset = ExpressionDataset(train_df)
val_dataset = ExpressionDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=config.training.shuffle, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_batch)


num_classes = df['result'].max() - df['result'].min() + 1

model_random = TransformerDecoder(
    num_tokens=config.model.num_tokens,
    n_embd=config.model.embedding_dim,
    num_layers=config.model.num_layers,
    num_heads=config.model.num_heads,
    num_classes=num_classes
).to(device)

optimizer_random = optim.Adam(model_random.parameters(), lr=config.training.learning_rate)
scheduler_random = transformers.get_linear_schedule_with_warmup(optimizer_random, (config.training.num_epochs + 9) // 10, config.training.num_epochs)
criterion_random = nn.CrossEntropyLoss()

random_train_losses, random_val_losses, random_train_accs, random_val_accs = train_learning(
    model_random, train_loader, val_loader, optimizer_random, criterion_random, scheduler_random, device, config.training.num_epochs
)


def operation_priority(expression):
    if '+' in expression or '-' in expression:
        return 1
    elif '*' in expression or '^' in expression:
        return 2
    return 3

df['priority'] = df['expression'].apply(operation_priority)
df = df.sort_values(by='priority')
df = df.drop(columns=['priority'])

train_df, val_df = train_test_split(df, test_size=config.data.train_split, random_state=config.data.random_seed)
train_dataset = ExpressionDataset(train_df)
val_dataset = ExpressionDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, collate_fn=collate_batch)

model_curriculum = TransformerDecoder(
    num_tokens=config.model.num_tokens,
    n_embd=config.model.embedding_dim,
    num_layers=config.model.num_layers,
    num_heads=config.model.num_heads,
    num_classes=num_classes
).to(device)

optimizer_curriculum = optim.Adam(model_curriculum.parameters(), lr=config.training.learning_rate)
scheduler_curriculum = transformers.get_linear_schedule_with_warmup(optimizer_curriculum, (config.training.num_epochs + 9) // 10, config.training.num_epochs)
criterion_curriculum = nn.CrossEntropyLoss()

curriculum_train_losses, curriculum_val_losses, curriculum_train_accs, curriculum_val_accs = train_learning(
    model_curriculum, train_loader, val_loader, optimizer_curriculum, criterion_curriculum, scheduler_curriculum, device, config.training.num_epochs
)

log_final_plots(
    config.training.num_epochs,
    random_train_losses,
    curriculum_train_losses,
    random_train_accs,
    curriculum_train_accs
)

writer.close()

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(range(1, config.training.num_epochs+1), random_train_losses, label='Random Train Loss')
plt.plot(range(1, config.training.num_epochs+1), random_val_losses, label='Random Val Loss')
plt.plot(range(1, config.training.num_epochs+1), curriculum_train_losses, label='Curriculum Train Loss')
plt.plot(range(1, config.training.num_epochs+1), curriculum_val_losses, label='Curriculum Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Comparison')

plt.subplot(122)
plt.plot(range(1, config.training.num_epochs+1), random_train_accs, label='Random Train Accuracy')
plt.plot(range(1, config.training.num_epochs+1), random_val_accs, label='Random Val Accuracy')
plt.plot(range(1, config.training.num_epochs+1), curriculum_train_accs, label='Curriculum Train Accuracy')
plt.plot(range(1, config.training.num_epochs+1), curriculum_val_accs, label='Curriculum Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy Comparison')

plt.show()
