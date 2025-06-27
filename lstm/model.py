import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# custom dataset for variable-length network flows
class NetworkFlowDataset(Dataset):    
    def __init__(self, X, y, lengths):
        self.X = X
        self.y = y
        self.lengths = lengths
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]]), self.lengths[idx]

# custom collate function to handle variable-length sequences
def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    
    # sort by length (descending) for efficient packing
    sorted_data = sorted(zip(sequences, labels, lengths), key=lambda x: x[2], reverse=True)
    sequences, labels, lengths = zip(*sorted_data)
    
    # pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.cat(labels)
    lengths = torch.LongTensor(lengths)
    
    return padded_sequences, labels, lengths

class LSTM(nn.Module):
    def __init__(self, input_size=400, hidden_size=128, num_layers=2, num_classes=4, dropout=0.3):
        super(LSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # attention mechanism for early detection
        self.attention = nn.Linear(hidden_size * 2, 1)
        
        # classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x, lengths):
        batch_size = x.size(0)
        
        # pack padded sequences
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_x)
        
        # unpack sequences
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # apply attention mechanism for early detection
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        attended_output = torch.sum(attention_weights * lstm_output, dim=1)
        
        # classification
        x = self.dropout(attended_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output, attention_weights

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, class_weights=None):
    
    # loss function with weights if no weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_balanced_accuracies = []
    
    best_balanced_acc = 0.0
    patience_counter = 0
    
    # training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for sequences, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences, labels = sequences.to(device), labels.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for sequences, labels, lengths in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                lengths = lengths.to(device)
                
                outputs, _ = model(sequences, lengths)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_batches += 1
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # calculate metrics
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        val_accuracy = accuracy_score(all_labels, all_predictions)
        balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        val_balanced_accuracies.append(balanced_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
        print("-" * 50)
        
        # learning rate scheduling based on balanced accuracy
        scheduler.step(balanced_acc)
        
        # early stopping based on balanced accuracy
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best model saved! Balanced Accuracy: {balanced_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("Early stopping triggered!")
                break
    
    return train_losses, val_losses, val_accuracies, val_balanced_accuracies

def evaluate_model(model, test_loader, class_names=['Normal', 'Brute Force', 'XSS', 'SQL Injection']):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            lengths = lengths.to(device)
            
            outputs, _ = model(sequences, lengths)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(len(class_names))
    )
    
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")
    print()
    
    print("Per-Class Metrics:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            print(f"{class_name:15} - P: {precision_per_class[i]:.3f}, R: {recall_per_class[i]:.3f}, F1: {f1_per_class[i]:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    with open("data/X_varlen.pkl", "rb") as f:
        X = pickle.load(f)
    with open("data/y_varlen.pkl", "rb") as f:
        y = pickle.load(f)
    with open("data/lengths.pkl", "rb") as f:
        lengths = pickle.load(f)
    
    print(f"Loaded {len(X)} flows")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Train/test split (70:30)
    X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
        X, y, lengths, test_size=0.3, random_state=42, stratify=y
    )
    
    # Further split training data for validation (80:20 of training data)
    X_train, X_val, y_train, y_val, len_train, len_val = train_test_split(
        X_train, y_train, len_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    print(f"Class weights: {class_weights}")
    
    # Create datasets and data loaders
    train_dataset = NetworkFlowDataset(X_train, y_train, len_train)
    val_dataset = NetworkFlowDataset(X_val, y_val, len_val)
    test_dataset = NetworkFlowDataset(X_test, y_test, len_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    model = LSTM(
        input_size=400,  # MAX_PAYLOAD_LEN from preprocessing
        hidden_size=128,
        num_layers=2,
        num_classes=4,   # Normal, Brute Force, XSS, SQL Injection
        dropout=0.3
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses, val_accuracies, val_balanced_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=50, learning_rate=0.001, 
        class_weights=class_weights
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader)
    
    # Plot training curves
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 4, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 4, 3)
    plt.plot(val_balanced_accuracies, label='Validation Balanced Accuracy')
    plt.title('Validation Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    
    plt.subplot(1, 4, 4)
    class_counts = np.bincount(y)
    plt.bar(['Normal', 'Brute Force', 'XSS', 'SQL Injection'], class_counts)
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()