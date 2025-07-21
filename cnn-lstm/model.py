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

class CNNLSTM(nn.Module):
    def __init__(self, input_size=400, cnn_channels=[64, 128, 256], kernel_sizes=[3, 5, 7], 
                 pool_size=2, lstm_hidden=128, lstm_layers=2, num_classes=4, dropout=0.3):
        super(CNNLSTM, self).__init__()
        
        self.input_size = input_size
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.num_classes = num_classes
        
        # CNN feature extraction layers
        self.cnn_layers = nn.ModuleList()
        
        # First CNN layer (input channels = 1 for raw bytes)
        self.cnn_layers.append(
            nn.Sequential(
                nn.Conv1d(1, cnn_channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
                nn.BatchNorm1d(cnn_channels[0]),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=pool_size),
                nn.Dropout(dropout * 0.5)
            )
        )
        
        # Additional CNN layers
        for i in range(1, len(cnn_channels)):
            self.cnn_layers.append(
                nn.Sequential(
                    nn.Conv1d(cnn_channels[i-1], cnn_channels[i], kernel_size=kernel_sizes[i % len(kernel_sizes)], 
                             padding=kernel_sizes[i % len(kernel_sizes)]//2),
                    nn.BatchNorm1d(cnn_channels[i]),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=pool_size),
                    nn.Dropout(dropout * 0.5)
                )
            )
        
        # Calculate the output size after CNN layers
        self.cnn_output_size = cnn_channels[-1]
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention mechanism for better sequence modeling
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Global attention for early detection
        self.global_attention = nn.Linear(lstm_hidden * 2, 1)
        
        # Classification layers with residual connection
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_hidden * 2, lstm_hidden)
        self.fc2 = nn.Linear(lstm_hidden, lstm_hidden // 2)
        self.fc3 = nn.Linear(lstm_hidden // 2, num_classes)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(lstm_hidden * 2)
        
        # Residual connection
        self.residual = nn.Linear(lstm_hidden * 2, lstm_hidden // 2)
        
    def forward(self, x, lengths):
        batch_size, seq_len, feature_size = x.size()
        
        # Reshape for CNN: (batch_size * seq_len, 1, feature_size)
        x_cnn = x.view(batch_size * seq_len, 1, feature_size)
        
        # Apply CNN layers for local feature extraction
        cnn_features = x_cnn
        for cnn_layer in self.cnn_layers:
            cnn_features = cnn_layer(cnn_features)
        
        # Calculate new sequence length after pooling
        new_feature_size = cnn_features.size(-1)
        cnn_output_channels = cnn_features.size(1)
        
        # Reshape back: (batch_size, seq_len, cnn_output_channels * new_feature_size)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # If the CNN output is too large, apply adaptive pooling
        if cnn_features.size(-1) > self.cnn_output_size:
            # Apply global average pooling to reduce dimensionality
            cnn_features = cnn_features.view(batch_size, seq_len, cnn_output_channels, new_feature_size)
            cnn_features = torch.mean(cnn_features, dim=-1)  # Global average pooling
        
        # Ensure the feature size matches LSTM input size
        if cnn_features.size(-1) != self.cnn_output_size:
            # Use a linear layer to match dimensions
            if not hasattr(self, 'feature_adapter'):
                self.feature_adapter = nn.Linear(cnn_features.size(-1), self.cnn_output_size).to(device)
            cnn_features = self.feature_adapter(cnn_features)
        
        # Pack padded sequences for LSTM
        packed_cnn = pack_padded_sequence(cnn_features, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # LSTM forward pass
        packed_lstm_out, (hidden, cell) = self.lstm(packed_cnn)
        
        # Unpack sequences
        lstm_out, unpacked_lengths = pad_packed_sequence(packed_lstm_out, batch_first=True)
        
        # Apply layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Multi-head self-attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global attention for sequence-level representation
        global_attn_weights = torch.softmax(self.global_attention(attn_out), dim=1)
        sequence_repr = torch.sum(global_attn_weights * attn_out, dim=1)
        
        # Classification with residual connection
        x = self.dropout(sequence_repr)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Residual connection
        residual = self.residual(sequence_repr)
        x = self.relu(self.fc2(x))
        x = x + residual  # Add residual connection
        x = self.dropout(x)
        
        output = self.fc3(x)
        
        return output, global_attn_weights

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001, class_weights=None):
    
    # loss function with weights if no weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Use AdamW optimizer for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
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
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 50)
        
        # learning rate scheduling
        scheduler.step()
        
        # early stopping based on balanced accuracy
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_cnn_lstm_model.pth')
            print(f"New best model saved! Balanced Accuracy: {balanced_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 15:  # Increased patience for CNN-LSTM
                print("Early stopping triggered!")
                break
    
    return train_losses, val_losses, val_accuracies, val_balanced_accuracies

def evaluate_model(model, test_loader, class_names=['Normal', 'Brute Force', 'XSS', 'SQL Injection']):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            lengths = lengths.to(device)
            
            outputs, _ = model(sequences, lengths)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # calculate per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(len(class_names))
    )
    
    print("=" * 60)
    print("CNN-LSTM MODEL EVALUATION RESULTS")
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
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('CNN-LSTM Confusion Matrix')
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
        'probabilities': all_probabilities
    }

def main():
    # Load preprocessed data
    print("Loading preprocessed data...")
    with open("../data/X_varlen.pkl", "rb") as f:
        X = pickle.load(f)
    with open("../data/y_varlen.pkl", "rb") as f:
        y = pickle.load(f)
    with open("../data/lengths.pkl", "rb") as f:
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
    
    # Smaller batch size for CNN-LSTM due to higher memory usage
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # Initialize CNN-LSTM model
    model = CNNLSTM(
        input_size=400,  # MAX_PAYLOAD_LEN from preprocessing
        cnn_channels=[32, 64, 128],  # Progressive channel increase
        kernel_sizes=[3, 5, 7],      # Multi-scale feature extraction
        pool_size=2,
        lstm_hidden=128,
        lstm_layers=2,
        num_classes=4,   # Normal, Brute Force, XSS, SQL Injection
        dropout=0.4      # Slightly higher dropout for regularization
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {total_params:,} parameters ({trainable_params:,} trainable)")
    
    # Train the model
    print("Starting training...")
    train_losses, val_losses, val_accuracies, val_balanced_accuracies = train_model(
        model, train_loader, val_loader, 
        num_epochs=60,  # More epochs for CNN-LSTM
        learning_rate=0.0005,  # Slightly higher learning rate
        class_weights=class_weights
    )
    
    # Load best model
    model.load_state_dict(torch.load('best_cnn_lstm_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating CNN-LSTM model on test set...")
    results = evaluate_model(model, test_loader)
    
    # Plot training curves
    plt.figure(figsize=(20, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('CNN-LSTM Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.title('CNN-LSTM Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(val_balanced_accuracies, label='Validation Balanced Accuracy', color='purple')
    plt.title('CNN-LSTM Validation Balanced Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    class_counts = np.bincount(y)
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon']
    plt.bar(['Normal', 'Brute Force', 'XSS', 'SQL Injection'], class_counts, color=colors)
    plt.title('Class Distribution')
    plt.ylabel('Count')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    # Save final results
    results_summary = {
        'model_type': 'CNN-LSTM Hybrid',
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'final_results': results,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_balanced_accuracies': val_balanced_accuracies
        }
    }
    
    with open('cnn_lstm_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    
    print(f"\nResults saved to 'cnn_lstm_results.pkl'")
    print(f"Best model saved to 'best_cnn_lstm_model.pth'")

if __name__ == "__main__":
    main()