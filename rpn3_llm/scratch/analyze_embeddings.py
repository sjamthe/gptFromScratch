import os
import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import sys

# Update path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import RPNTokenizer

def load_checkpoint(path, device='cpu'):
    print(f"Loading checkpoint {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    # The embeddings are usually stored in 'transformer.wte.weight'
    wte = checkpoint['model']['transformer.wte.weight'].float().cpu().numpy()
    return wte

def get_vocab():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(base_dir, "rpn-tokenizer.json"), "r") as f:
        data = json.load(f)
    vocab = data["model"]["vocab"]
    return vocab

def main():
    model_path = "../models/ut1.5M_2l_8h_384e_mlp3_phaseMask_True_rpn3_208000.pt"
    if not os.path.exists(model_path):
        print(f"Error: Could not find checkpoint {model_path}")
        return

    embeddings = load_checkpoint(model_path)
    vocab = get_vocab()
    
    # Reverse vocab to map ID -> Token
    id_to_token = {v: k for k, v in vocab.items()}
    num_tokens = len(vocab)
    
    # 1. Similarity Matrix of Digits
    digits = [str(i) for i in range(10)]
    digit_ids = [vocab[d] for d in digits]
    digit_embs = embeddings[digit_ids]
    
    sim_matrix = cosine_similarity(digit_embs)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, xticklabels=digits, yticklabels=digits, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Cosine Similarity: Digits 0-9")
    plt.savefig("digit_similarity.png")
    plt.close()
    print("Saved digit_similarity.png")

    # 2. PCA of Digits
    pca = PCA(n_components=2)
    digit_pca = pca.fit_transform(digit_embs)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(digit_pca[:, 0], digit_pca[:, 1], c='blue', marker='o')
    for i, d in enumerate(digits):
        plt.annotate(d, (digit_pca[i, 0], digit_pca[i, 1]), fontsize=16, ha='right')
    plt.title("PCA of Digits 0-9")
    plt.grid(True)
    plt.savefig("digit_pca.png")
    plt.close()
    print("Saved digit_pca.png")
    
    # 3. PCA of Structural Tokens vs Digits
    structural_tokens = ["[BOS]", "[EOS]", "[REV]", "[MATH]", "[SEP]", "[ANS]", "+", "-", "=", "[BORROW]"]
    struct_ids = [vocab[t] for t in structural_tokens]
    struct_embs = embeddings[struct_ids]
    
    combined_embs = np.vstack([digit_embs, struct_embs])
    combined_labels = digits + structural_tokens
    
    pca2 = PCA(n_components=2)
    combined_pca = pca2.fit_transform(combined_embs)
    
    plt.figure(figsize=(12, 10))
    # Plot digits
    plt.scatter(combined_pca[:10, 0], combined_pca[:10, 1], c='blue', marker='o', label="Digits", s=100)
    # Plot structure
    plt.scatter(combined_pca[10:, 0], combined_pca[10:, 1], c='red', marker='x', label="Structural", s=100)
    
    for i, label in enumerate(combined_labels):
        plt.annotate(label, (combined_pca[i, 0], combined_pca[i, 1]), fontsize=12)
        
    plt.title("PCA: Digits vs Structural Tokens")
    plt.legend()
    plt.grid(True)
    plt.savefig("structural_pca.png")
    plt.close()
    print("Saved structural_pca.png")

    # 4. Algebraic Tests: Emb(a) + Emb(b) ?
    def find_closest(vec, embeddings, top_k=3):
        sims = cosine_similarity(vec.reshape(1, -1), embeddings)[0]
        closest_ids = np.argsort(sims)[::-1][:top_k]
        return [(id_to_token[i], sims[i]) for i in closest_ids]

    print("\n--- Algebraic Tests ---")
    
    # Test 1: Addition without carry
    print("\nTest: Emb('2') + Emb('3')")
    res1 = embeddings[vocab['2']] + embeddings[vocab['3']]
    closest1 = find_closest(res1, embeddings)
    for tok, sim in closest1: print(f"  {tok}: {sim:.3f}")

    # Test 2: Addition with carry
    print("\nTest: Emb('7') + Emb('5')")
    res2 = embeddings[vocab['7']] + embeddings[vocab['5']]
    closest2 = find_closest(res2, embeddings)
    for tok, sim in closest2: print(f"  {tok}: {sim:.3f}")

    # Test 3: Subtraction
    print("\nTest: Emb('9') - Emb('4')")
    res3 = embeddings[vocab['9']] - embeddings[vocab['4']]
    closest3 = find_closest(res3, embeddings)
    for tok, sim in closest3: print(f"  {tok}: {sim:.3f}")

    # Test 4: Operators
    print("\nTest: Emb('+') + Emb('-')")
    res4 = embeddings[vocab['+']] + embeddings[vocab['-']]
    closest4 = find_closest(res4, embeddings)
    for tok, sim in closest4: print(f"  {tok}: {sim:.3f}")

if __name__ == "__main__":
    main()
