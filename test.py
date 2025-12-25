
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import sys
import glob

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Model Classes (Copied from main.py)
# ==========================================

class AttentionModule(nn.Module):
    def __init__(self, cnn_feature_size, hidden_size, attention_size):
        super(AttentionModule, self).__init__()
        self.attention = nn.Linear(cnn_feature_size + hidden_size, attention_size)
        self.attention_score = nn.Linear(attention_size, 1)

    def forward(self, cnn_features, hidden_state):
        batch_size, channels, H, W = cnn_features.size()
        cnn_features_flat = cnn_features.view(batch_size, channels, -1).permute(0, 2, 1)
        num_pixels = H * W
        hidden_expanded = hidden_state.unsqueeze(1).repeat(1, num_pixels, 1)
        combined = torch.cat([cnn_features_flat, hidden_expanded], dim=2)
        attention_hidden = torch.tanh(self.attention(combined))
        attention_scores = self.attention_score(attention_hidden).squeeze(2)
        attention_weights = torch.softmax(attention_scores, dim=1)
        context_vector = torch.sum(attention_weights.unsqueeze(2) * cnn_features_flat, dim=1)
        return context_vector, attention_weights

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, attention_module, attention_size):
        super(LSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_module = attention_module
        combined_size = input_size + hidden_size + attention_size
        self.W_i = nn.Linear(combined_size, hidden_size)
        self.W_f = nn.Linear(combined_size, hidden_size)
        self.W_c = nn.Linear(combined_size, hidden_size)
        self.W_o = nn.Linear(combined_size, hidden_size)

    def forward(self, x, cnn_features, hidden_state=None):
        batch_size, seq_len, _ = x.size()
        if hidden_state is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = hidden_state
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            context_vector, _ = self.attention_module(cnn_features, h_t)
            combined = torch.cat([x_t, h_t, context_vector], dim=1)
            i_t = torch.sigmoid(self.W_i(combined))
            f_t = torch.sigmoid(self.W_f(combined))
            o_t = torch.sigmoid(self.W_o(combined))
            c_tilde = torch.tanh(self.W_c(combined))
            c_t = f_t * c_t + i_t * c_tilde
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t.unsqueeze(1))
        
        outputs = torch.cat(outputs, dim=1)
        return outputs, (h_t, c_t)

class ImageCaptioningModel(nn.Module):
    def __init__(self, cnn_model, embedding_model, lstm_hidden_dim, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.cnn_model = cnn_model
        self.embedding = embedding_model
        embed_dim = embedding_model.weight.shape[1]
        cnn_feature_size = 2048
        attention_size = 512
        self.attention = AttentionModule(cnn_feature_size, lstm_hidden_dim, attention_size)
        self.lstm = LSTMWithAttention(embed_dim, lstm_hidden_dim, self.attention, cnn_feature_size)
        self.fc = nn.Linear(lstm_hidden_dim, vocab_size)

    def forward(self, images, captions):
        with torch.no_grad():
            cnn_features = self.cnn_model(images)
        embedded = self.embedding(captions)
        lstm_out, _ = self.lstm(embedded, cnn_features)
        outputs = self.fc(lstm_out)
        return outputs

    def generate_caption(self, image, max_length=20, start_token=1, end_token=2):
        self.eval()
        with torch.no_grad():
            cnn_features = self.cnn_model(image.unsqueeze(0))
            caption = [start_token]
            for _ in range(max_length - 1):
                caption_tensor = torch.LongTensor([caption]).to(image.device)
                embedded = self.embedding(caption_tensor)
                lstm_out, _ = self.lstm(embedded, cnn_features)
                outputs = self.fc(lstm_out)
                next_token = outputs[0, -1, :].argmax().item()
                caption.append(next_token)
                if next_token == end_token:
                    break
        return caption

# ==========================================
# Helpers
# ==========================================

def reassemble_model(output_path="best_model.pth", parts_pattern="model_parts/best_model.pth.part_*"):
    """Reassemble split model parts into the full model file."""
    parts = sorted(glob.glob(parts_pattern))
    if not parts:
        return False
        
    print(f"Reassembling model from {len(parts)} parts...")
    with open(output_path, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    print(f"✓ Model reassembled at {output_path}")
    return True

def untokenize(tokens, idx2word):
    res = []
    for x in tokens:
        if x < len(idx2word):
            word = idx2word[x]
            if word not in ["<PAD>", "<START>", "<END>"]:
                res.append(word)
    return ' '.join(res)

def load_trained_model(checkpoint_path, device):
    # Check if model exists, if not try to reassemble
    if not os.path.exists(checkpoint_path):
        print(f"Model file {checkpoint_path} not found.")
        # Try to find parts
        if not reassemble_model(checkpoint_path, "model_parts/" + checkpoint_path + ".part_*"):
            print(f"Error: Could not find model or model parts for {checkpoint_path}")
            sys.exit(1)

    print(f"Loading model from {checkpoint_path}...")
    try:
        # Use weights_only=False because we are loading complex objects (classes, vocabs)
        # Note: In production/untrusted envs this has security implications
        checkpoint = torch.load(checkpoint_path, map_location=device) #, weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    # Load vocab
    if 'vocab' in checkpoint:
        idx2word, word2idx = checkpoint['vocab']
    else:
        print("Error: Vocab not found in checkpoint.")
        sys.exit(1)
    
    vocab_size = len(idx2word)
    lstm_hidden_dim = 256 # From main.py default

    # Recreate ResNet
    resnet = models.resnet50(weights=None) 
    resnet = nn.Sequential(*list(resnet.children())[:-2])
    
    # Recreate Embedding
    embedding = nn.Embedding(vocab_size, 300) 
    
    # Initialize Model
    model = ImageCaptioningModel(
        cnn_model=resnet,
        embedding_model=embedding,
        lstm_hidden_dim=lstm_hidden_dim,
        vocab_size=vocab_size
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully (Epoch {checkpoint.get('epoch', '?')})")
    
    return model, idx2word, word2idx

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate caption for an image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output_examples", help="Directory to save result")
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Model
    model, idx2word, word2idx = load_trained_model(args.model_path, device)
    
    # Load and Preprocess Image
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        sys.exit(1)
        
    try:
        image_pil = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_pil).to(device)
    
    # Generate Caption
    print("Generating caption...")
    tokens = model.generate_caption(
        image_tensor,
        start_token=word2idx.get("<START>", 1),
        end_token=word2idx.get("<END>", 2)
    )
    
    caption = untokenize(tokens, idx2word)
    print("\n" + "="*40)
    print(f"GENERATED CAPTION: {caption}")
    print("="*40 + "\n")
    
    # Save Result
    plt.figure(figsize=(10, 8))
    plt.imshow(image_pil)
    plt.axis("off")
    plt.title(f"Generated: {caption}", fontsize=14, wrap=True)
    
    filename = os.path.basename(args.image_path)
    save_path = os.path.join(args.output_dir, f"test_{filename}")
    plt.savefig(save_path)
    plt.close()
    
    print(f"Result saved to: {save_path}")
