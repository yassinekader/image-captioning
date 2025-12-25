# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import numpy as np
import struct
from tqdm import tqdm
import os
from PIL import Image
import zipfile
import gzip
import shutil

# %% [markdown]
# # Download Dataset from Kaggle & Word2Vec Model

# %%
import kagglehub
path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")

# %%
!ls -la {path}/flickr30k_images/results.csv

# %%


# %%
!wget https://huggingface.co/NathaNn1111/word2vec-google-news-negative-300-bin/resolve/main/GoogleNews-vectors-negative300.bin

# %%
!ls

# %%
ROOT_DIR = f"{path}/flickr30k_images/flickr30k_images"
ANN_FILE = f"{path}/flickr30k_images/results.csv"
WORD2VEC_PATH = 'GoogleNews-vectors-negative300.bin'

# %% [markdown]
# # Create Custom Flickr30k Dataset

# %%
import pandas as pd

class Flickr30kDataset(Dataset):
    """Custom Dataset for Flickr30k with CSV annotations"""

    def __init__(self, root_dir, ann_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Read CSV file
        if os.path.exists(ann_file):
            self.annotations = pd.read_csv(ann_file, delimiter='|')
            # CSV format: image_name | comment_number | comment
            self.annotations.columns = ['image', 'comment_number', 'comment']
        else:
            # Try alternative format
            ann_file_alt = 'flickr30k/results.txt'
            if os.path.exists(ann_file_alt):
                self.annotations = pd.read_csv(ann_file_alt, delimiter='\t', header=None)
                self.annotations.columns = ['image', 'comment']
            else:
                raise FileNotFoundError(f"Could not find annotation file at {ann_file}")

        # Get unique images
        self.image_names = self.annotations['image'].unique()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # If image not found, return a blank image
            image = Image.new('RGB', (224, 224), color='white')

        # Get all captions for this image
        captions = self.annotations[self.annotations['image'] == img_name]['comment'].tolist()

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, captions

# %% [markdown]
# # Vocabulary and Word2Vec Functions

# %%
def read_until(open_file):
    """Read characters from file until space"""
    while True:
        char = open_file.read(1)
        if char == b' ' or char == b'':
            break
        yield char

# %%
def read_word2vec(word2vec_path, max_words=50000):
    """Generator to read Word2Vec binary file (limited for speed)"""
    try:
        with open(word2vec_path, "rb") as f:
            first_line = f.readline()
            n_words, n_dim = map(int, first_line.split())
            print(f"Word2Vec has {n_words} words, dimension {n_dim}")

            for n_word in range(min(n_words, max_words)):
                try:
                    word = b''.join(read_until(f)).decode('utf-8', errors='ignore')
                    vector = struct.unpack('f' * n_dim, f.read(n_dim * 4))
                    if len(vector) == n_dim:
                        yield word, vector
                except:
                    continue
    except Exception as e:
        print(f"Error reading Word2Vec: {e}")
        print("Falling back to random embeddings...")
        return

# %%
def create_embedding_layer(vocab, word2vec_path, randomized=False):
    """Create embedding layer from Word2Vec or random initialization"""
    if randomized or not os.path.exists(word2vec_path):
        print("Using random embeddings...")
        idx_to_word = ["<PAD>", "<START>", "<END>", "<UNK>"]
        word_to_idx = {w: i for i, w in enumerate(idx_to_word)}
        weights = [np.random.randn(300) * 0.01 for _ in range(len(idx_to_word))]

        for word in vocab:
            if word not in word_to_idx:
                idx_to_word.append(word)
                word_to_idx[word] = len(idx_to_word) - 1
                weights.append(np.random.randn(300) * 0.01)
    else:
        weights = []
        idx_to_word = ["<PAD>", "<START>", "<END>", "<UNK>"]
        word_to_idx = {w: i for i, w in enumerate(idx_to_word)}

        # Add random embeddings for special tokens
        for _ in range(4):
            weights.append(np.random.randn(300) * 0.01)

        print("Reading Word2Vec embeddings (this may take a minute)...")
        word_count = 0
        for word, vector in read_word2vec(word2vec_path, max_words=50000):
            if word.lower() in vocab:
                idx_to_word.append(word.lower())
                word_to_idx[word.lower()] = len(idx_to_word) - 1
                weights.append(np.array(vector))
                word_count += 1

        print(f"Loaded {word_count} words from Word2Vec")

        # Add random embeddings for words not in Word2Vec
        for word in vocab:
            if word not in word_to_idx:
                idx_to_word.append(word)
                word_to_idx[word] = len(idx_to_word) - 1
                weights.append(np.random.randn(300) * 0.01)

    n_words = len(weights)
    n_dim = 300
    weights = np.array(weights)

    emb_layer = nn.Embedding(n_words, n_dim)
    emb_layer.weight.data.copy_(torch.from_numpy(weights).float())

    if not randomized:
        emb_layer.weight.requires_grad = False

    return emb_layer, idx_to_word, word_to_idx

# %%
def tokenize(sentence, word2idx, max_length=20):
    """Convert sentence to token indices"""
    res = [word2idx.get("<START>", 1)]  # Start token
    sentence = sentence.lower().strip()

    for word in sentence.split():
        # Clean word
        word = ''.join(c for c in word if c.isalnum() or c == "'")
        if word in word2idx:
            res.append(word2idx[word])
        else:
            res.append(word2idx.get("<UNK>", 3))  # Unknown token

    res.append(word2idx.get("<END>", 2))  # End token

    if len(res) > max_length:
        return res[:max_length]
    return res + [word2idx.get("<PAD>", 0)] * (max_length - len(res))

def untokenize(tokens, idx2word):
    """Convert token indices back to words"""
    res = []
    for x in tokens:
        if x < len(idx2word):
            word = idx2word[x]
            if word not in ["<PAD>", "<START>", "<END>"]:
                res.append(word)
    return ' '.join(res)

# %% [markdown]
# # Load and Prepare Dataset
# 

# %%
vocabulary = set()
df = pd.read_csv(ANN_FILE, delimiter='|')
df.columns = ['image', 'comment_number', 'comment']

# %%
df

# %%
full_dataset = Flickr30kDataset(
    root_dir=ROOT_DIR,
    ann_file=ANN_FILE,
    transform=transform
)

# %%
captions_col = 'comment'

# %% [markdown]
# # Extract Vocabulary from Captions

# %%
for caption in df[captions_col].dropna():
    caption = str(caption).lower().strip()
    for word in caption.split():
        word = ''.join(c for c in word if c.isalnum() or c == "'")
        if word:
            vocabulary.add(word)

# %%
print(f"Vocabulary size: {len(vocabulary)} words")

# %% [markdown]
# # Define Image Transformations

# %%
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# %%
import random
import matplotlib.pyplot as plt

# Helper functions (copied to ensure they are available for plotting)
def denormalize_image(image_tensor):
    """
    Denormalize image for visualization
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.cpu() * std + mean
    image = torch.clamp(image, 0, 1)

    return image


def show_image_with_caption(image, generated_caption, actual_caption):
    """
    Display image with generated and actual captions
    """
    image_np = (
        denormalize_image(image)
        .permute(1, 2, 0)
        .numpy()
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(image_np)
    plt.axis("off")
    plt.title(
        f"Generated: {generated_caption}\n\nActual: {actual_caption}",
        fontsize=12,
        pad=20
    )
    plt.tight_layout()
    plt.show()


print("\n" + "=" * 60)
print("DISPLAYING SAMPLE IMAGES AND CAPTIONS FROM DATASET")
print("=" * 60 + "\n")

for i in range(5):
    idx = random.randint(0, len(full_dataset) - 1)
    image, captions = full_dataset[idx]
    actual_caption = random.choice(captions)
    show_image_with_caption(image, "", actual_caption)
    print("\n" + "=" * 60)
    print("-" * 60 + "\n")

# %%
print(f"Total images: {len(full_dataset)}")

# %%
# Split dataset 80/20
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size

# %%
train_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# %%
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# %% [markdown]
# # Load Pre-trained Embeddings

# %%
RANDOMIZED = True
pretrained_embeddings, idx2word, word2idx = create_embedding_layer(
    vocabulary,
    WORD2VEC_PATH,
    RANDOMIZED
)

# %%
print(f"Final vocabulary size: {len(idx2word)} tokens")
print(f"Special tokens: {idx2word[:4]}")

# %%
print("Loading ResNet50...")
resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')

# %%
print("\n=== Original ResNet50 (last layers) ===")
print(f"Average Pool: {resnet.avgpool}")
print(f"FC Layer: {resnet.fc}")

# %%
resnet = nn.Sequential(*list(resnet.children())[:-2])

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# I.7: Freeze all ResNet parameters
for param in resnet.parameters():
    param.requires_grad = False

resnet = resnet.to(device)
resnet.eval()

# %% [markdown]
# # Attention Module

# %%
class AttentionModule(nn.Module):
    """
    Attention module: computes context vector from CNN features and RNN hidden state

    Equations:
        attention_scores = W_att * concat(features, hidden_state)
        context_vector = sum(softmax(attention_scores) * features)
    """
    def __init__(self, cnn_feature_size, hidden_size, attention_size):
        super(AttentionModule, self).__init__()
        # Project concatenated features to attention space
        self.attention = nn.Linear(cnn_feature_size + hidden_size, attention_size)
        # Compute scalar attention score for each spatial location
        self.attention_score = nn.Linear(attention_size, 1)

    def forward(self, cnn_features, hidden_state):
        """
        Args:
            cnn_features: (batch, channels, H, W) - CNN feature maps
            hidden_state: (batch, hidden_size) - LSTM hidden state
        Returns:
            context_vector: (batch, channels) - weighted sum of features
            attention_weights: (batch, H*W) - attention distribution
        """
        batch_size, channels, H, W = cnn_features.size()

        # Flatten spatial dimensions: (batch, channels, H, W) -> (batch, H*W, channels)
        cnn_features_flat = cnn_features.view(batch_size, channels, -1).permute(0, 2, 1)
        num_pixels = H * W

        # Expand hidden state to match each spatial location
        # (batch, hidden) -> (batch, H*W, hidden)
        hidden_expanded = hidden_state.unsqueeze(1).repeat(1, num_pixels, 1)

        # Concatenate features and hidden state for each location
        # (batch, H*W, channels + hidden)
        combined = torch.cat([cnn_features_flat, hidden_expanded], dim=2)

        # Compute attention scores for each location
        attention_hidden = torch.tanh(self.attention(combined))  # (batch, H*W, att_size)
        attention_scores = self.attention_score(attention_hidden).squeeze(2)  # (batch, H*W)

        # Normalize with softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, H*W)

        # Compute weighted sum of features (context vector)
        # (batch, H*W, 1) * (batch, H*W, channels) -> sum -> (batch, channels)
        context_vector = torch.sum(
            attention_weights.unsqueeze(2) * cnn_features_flat,
            dim=1
        )

        return context_vector, attention_weights

# %% [markdown]
# # LSTM with Attention

# %%
class LSTMWithAttention(nn.Module):
    """
    Custom LSTM that incorporates attention at each time step

    Equations at each time step t:
        i_t = σ(W_i * concat(x_t, h_{t-1}, att_t))
        f_t = σ(W_f * concat(x_t, h_{t-1}, att_t))
        c_t = f_t * c_{t-1} + i_t * tanh(W_c * concat(x_t, h_{t-1}, att_t))
        o_t = σ(W_o * concat(x_t, h_{t-1}, att_t))
        h_t = o_t * tanh(c_t)
    """
    def __init__(self, input_size, hidden_size, attention_module, attention_size):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_module = attention_module

        # Input to gates: x_t + h_{t-1} + attention_context
        combined_size = input_size + hidden_size + attention_size

        # LSTM gates (following the equations from TP)
        self.W_i = nn.Linear(combined_size, hidden_size)  # Input gate
        self.W_f = nn.Linear(combined_size, hidden_size)  # Forget gate
        self.W_c = nn.Linear(combined_size, hidden_size)  # Cell gate
        self.W_o = nn.Linear(combined_size, hidden_size)  # Output gate

    def forward(self, x, cnn_features, hidden_state=None):
        """
        Args:
            x: (batch, seq_len, input_size) - embedded captions
            cnn_features: (batch, channels, H, W) - CNN features
            hidden_state: tuple (h, c) or None - initial hidden/cell state
        Returns:
            outputs: (batch, seq_len, hidden_size) - LSTM outputs
            (h_n, c_n): final hidden and cell states
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden state if not provided
        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = hidden_state

        outputs = []

        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]  # Current input (batch, input_size)

            # Get attention context from current hidden state
            context_vector, _ = self.attention_module(cnn_features, h_t)

            # Concatenate: input + previous hidden + attention context
            combined = torch.cat([x_t, h_t, context_vector], dim=1)

            # Compute LSTM gates (following TP equations)
            i_t = torch.sigmoid(self.W_i(combined))  # Input gate
            f_t = torch.sigmoid(self.W_f(combined))  # Forget gate
            o_t = torch.sigmoid(self.W_o(combined))  # Output gate
            c_tilde = torch.tanh(self.W_c(combined))  # Cell candidate

            # Update cell state: c_t = f_t * c_{t-1} + i_t * c_tilde
            c_t = f_t * c_t + i_t * c_tilde

            # Update hidden state: h_t = o_t * tanh(c_t)
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(1))

        # Concatenate all time steps
        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, hidden_size)

        return outputs, (h_t, c_t)

# %% [markdown]
# # Complete Image Captioning Model

# %%
class ImageCaptioningModel(nn.Module):
    """
    Complete image captioning model

    Architecture:
        1. CNN (ResNet50) extracts visual features
        2. Embeddings convert word tokens to vectors
        3. LSTM with Attention generates caption word by word
        4. FC layer predicts next word probabilities
    """
    def __init__(self, cnn_model, embedding_model, lstm_hidden_dim, vocab_size):
        super(ImageCaptioningModel, self).__init__()

        self.cnn_model = cnn_model
        self.embedding = embedding_model

        # Get dimensions
        embed_dim = embedding_model.weight.shape[1]  # 300 for Word2Vec
        cnn_feature_size = 2048  # ResNet50 output channels
        attention_size = 512  # Attention projection dimension

        # Create attention module
        self.attention = AttentionModule(
            cnn_feature_size,
            lstm_hidden_dim,
            attention_size
        )

        # Create LSTM with attention
        self.lstm = LSTMWithAttention(
            embed_dim,
            lstm_hidden_dim,
            self.attention,
            cnn_feature_size  # Context vector size
        )

        # Output layer to predict vocabulary
        self.fc = nn.Linear(lstm_hidden_dim, vocab_size)

    def forward(self, images, captions):
        """
        Training forward pass

        Args:
            images: (batch, 3, 224, 224) - input images
            captions: (batch, seq_len) - tokenized captions
        Returns:
            outputs: (batch, seq_len, vocab_size) - word predictions
        """
        # Extract CNN features (frozen, no gradients)
        with torch.no_grad():
            cnn_features = self.cnn_model(images)  # (batch, 2048, 7, 7)

        # Embed caption tokens
        embedded = self.embedding(captions)  # (batch, seq_len, 300)

        # Process through LSTM with attention
        lstm_out, _ = self.lstm(embedded, cnn_features)  # (batch, seq_len, hidden_dim)

        # Predict word probabilities
        outputs = self.fc(lstm_out)  # (batch, seq_len, vocab_size)

        return outputs

    def generate_caption(self, image, max_length=20, start_token=1, end_token=2):
        """
        Generate caption for a single image (inference)

        Args:
            image: (3, 224, 224) - single image tensor
            max_length: maximum caption length
            start_token: index of <START> token
            end_token: index of <END> token
        Returns:
            caption: list of token indices
        """
        self.eval()
        with torch.no_grad():
            # Extract CNN features
            cnn_features = self.cnn_model(image.unsqueeze(0))

            # Start with <START> token
            caption = [start_token]

            for _ in range(max_length - 1):
                # Embed current caption
                caption_tensor = torch.LongTensor([caption]).to(image.device)
                embedded = self.embedding(caption_tensor)

                # Get LSTM output for current sequence
                lstm_out, _ = self.lstm(embedded, cnn_features)

                # Predict next word from last time step
                output = self.fc(lstm_out[:, -1, :])

                # Get most likely word
                predicted = output.argmax(1).item()
                caption.append(predicted)

                # Stop if <END> token generated
                if predicted == end_token:
                    break

        return caption

# %% [markdown]
# # Initialize Model

# %%
# Optimized hyperparameters for T4 GPU
LSTM_HIDDEN_DIM = 256  # Reduced for faster training
VOCAB_SIZE = len(idx2word)

model = ImageCaptioningModel(
    cnn_model=resnet,
    embedding_model=pretrained_embeddings,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
    vocab_size=VOCAB_SIZE
).to(device)

# %%
# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

# %%
print(f"\n=== Model Statistics ===")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")
print(f"Model size: ~{trainable_params * 4 / 1024 / 1024:.2f} MB")

# %% [markdown]
# # Prepare DataLoaders with Collate Function

# %%
MAX_CAPTION_LENGTH = 20

def collate_fn(batch):
    """
    Custom collate function to handle multiple captions per image

    Args:
        batch: list of (image, captions) tuples
    Returns:
        images: (batch, 3, 224, 224) tensor
        captions: (batch, max_len) tensor of tokenized captions
    """
    images = []
    captions = []

    for img, caption_list in batch:
        images.append(img)

        # Take first caption if multiple available
        caption_text = caption_list[0] if isinstance(caption_list, list) else caption_list

        # Tokenize caption
        tokenized = tokenize(caption_text, word2idx, MAX_CAPTION_LENGTH)
        captions.append(tokenized)

    images = torch.stack(images)
    captions = torch.LongTensor(captions)

    return images, captions

# %%
# Create DataLoaders
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True  # Faster GPU transfer
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
    pin_memory=True
)

# %%
print(f"✓ DataLoaders created")
print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

# %% [markdown]
# # Training Loop Setup

# %%
# Hyperparameters (optimized for T4 GPU and reasonable training time)
LEARNING_RATE = 0.001
NUM_EPOCHS = 15  
LOG_INTERVAL = 10 

# Loss function (ignore padding tokens)
criterion = nn.CrossEntropyLoss(ignore_index=word2idx.get("<PAD>", 0))

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler (StepLR as required by TP)
scheduler = optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,  # Reduce LR every 5 epochs
    gamma=0.5      # Multiply LR by 0.5
)

# %%
writer = SummaryWriter('runs/image_captioning')

print("=== Training Configuration ===")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Scheduler: StepLR (step=5, gamma=0.5)")
print(f"Optimizer: Adam")
print(f"Loss function: CrossEntropyLoss (ignore_index={word2idx.get('<PAD>', 0)})")


# %%
global_step = 0
best_loss = float('inf')

for epoch in range(NUM_EPOCHS):
    # ========== TRAINING ==========
    model.train()
    epoch_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

    for batch_idx, (images, captions) in enumerate(pbar):
        # Move to device
        images = images.to(device)
        # Captions for training: input sequence (excluding last token) and target sequence (excluding first token)
        # Input to LSTM should be <START>, word1, word2, ..., wordN
        # Target for loss should be word1, word2, ..., wordN, <END>
        captions_input = captions[:, :-1].to(device)  # Input to LSTM (e.g., <START>, w1, w2, ..., wN)
        captions_target = captions[:, 1:].to(device) # Target for loss (e.g., w1, w2, ..., wN, <END>)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images, captions_input) # model now takes captions_input

        # Reshape for loss calculation
        # outputs: (batch_size, seq_len-1, vocab_size) -> (batch_size * (seq_len-1), vocab_size)
        outputs_flat = outputs.contiguous().view(-1, VOCAB_SIZE)
        # captions_target: (batch_size, seq_len-1) -> (batch_size * (seq_len-1))
        targets_flat = captions_target.contiguous().view(-1)

        # Calculate loss
        loss = criterion(outputs_flat, targets_flat)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Update statistics
        epoch_loss += loss.item()
        num_batches += 1

        # Logging
        if batch_idx % LOG_INTERVAL == 0:
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        global_step += 1

    # Average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    writer.add_scalar('Loss/train_epoch', avg_epoch_loss, epoch)

    print(f'\n[Epoch {epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_epoch_loss:.4f}')

    # ========== EVALUATION ==========
    model.eval()
    with torch.no_grad():
        test_iter = iter(test_loader)
        sample_images, sample_captions = next(test_iter)
        sample_image = sample_images[0].to(device)

        generated_tokens = model.generate_caption(
            sample_image,
            max_length=MAX_CAPTION_LENGTH,
            start_token=word2idx.get("<START>", 1),
            end_token=word2idx.get("<END>", 2)
        )

        generated_caption = untokenize(generated_tokens, idx2word)
        actual_caption = untokenize(sample_captions[0].tolist(), idx2word)

        print("\n--- Caption Generation Example ---")
        print(f"Generated: {generated_caption}")
        print(f"Actual:    {actual_caption}")
        print("-" * 50 + "\n")

        writer.add_text('Captions/Generated', generated_caption, epoch)
        writer.add_text('Captions/Actual', actual_caption, epoch)

    # ========== SCHEDULER ==========
    old_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    new_lr = scheduler.get_last_lr()[0]

    if old_lr != new_lr:
        print(f"Learning rate changed: {old_lr:.6f} -> {new_lr:.6f}")

    writer.add_scalar('Learning_Rate', new_lr, epoch)

    # Save best model
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': best_loss,
            'vocab': (idx2word, word2idx)
        }, 'best_model.pth')
        print(f"✓ Best model saved (loss: {best_loss:.4f})")

    print("=" * 60)

writer.close()
print("\n🎉 TRAINING COMPLETED! 🎉\n")

# %%
# !mkdir /content/drive/MyDrive/Genai

# %%
torch.save({
    'epoch': NUM_EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'vocab': (idx2word, word2idx),
    'config': {
      'lstm_hidden_dim': LSTM_HIDDEN_DIM,
      'vocab_size': VOCAB_SIZE,
      'max_caption_length': MAX_CAPTION_LENGTH
    }
}, '/content/drive/MyDrive/Genai/final_model.pth')

# %% [markdown]
# # Testing Using The Exported Model

# %%
def load_trained_model(checkpoint_path="best_model.pth"):
    """
    Load a saved model checkpoint
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Recreate model architecture
    loaded_model = ImageCaptioningModel(
        cnn_model=resnet,
        embedding_model=pretrained_embeddings,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        vocab_size=VOCAB_SIZE
    ).to(device)

    # Load weights
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    print(f"✓ Model loaded from epoch {checkpoint['epoch']}")

    if "loss" in checkpoint:
        print(f"  Loss: {checkpoint['loss']:.4f}")

    return loaded_model

# %%
import matplotlib.pyplot as plt

def show_image_with_caption(image, generated_caption, actual_caption):
    """
    Display image with generated and actual captions
    """
    image_np = (
        denormalize_image(image)
        .permute(1, 2, 0)
        .numpy()
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(image_np)
    plt.axis("off")
    plt.title(
        f"Generated: {generated_caption}\n\nActual: {actual_caption}",
        fontsize=12,
        pad=20
    )
    plt.tight_layout()
    plt.show()

# %%
def denormalize_image(image_tensor):
    """
    Denormalize image for visualization
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.cpu() * std + mean
    image = torch.clamp(image, 0, 1)

    return image

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Re-defining these variables to ensure they are available for load_trained_model
LSTM_HIDDEN_DIM = 256
VOCAB_SIZE = len(idx2word)

testing_model = load_trained_model("best_model.pth")
testing_model.eval()
test_iter = iter(test_loader)
images, captions = next(test_iter)

num_examples = min(5, len(images))

for i in range(num_examples):
    image = images[i].to(device)

    with torch.no_grad():
        generated_tokens = testing_model.generate_caption(
            image=image,
            max_length=MAX_CAPTION_LENGTH,
            start_token=word2idx.get("<START>", 1),
            end_token=word2idx.get("<END>", 2)
        )

    generated_caption = untokenize(generated_tokens, idx2word)
    print(generated_caption)
    actual_caption = untokenize(captions[i].tolist(), idx2word)

    show_image_with_caption(
        images[i],
        generated_caption,
        actual_caption
    )


