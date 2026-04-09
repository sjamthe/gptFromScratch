from train_gpt2 import DataLoaderLite
import torch
import tiktoken

def analyze_dataset():
    print("Loading dataset...")
    # Initialize DataLoaderLite from train_gpt2
    # B and T values don't matter since we are only accessing all loaded tokens
    loader = DataLoaderLite(B=1, T=1)
    
    tokens = loader.tokens
    
    print("Counting token frequencies...")
    # Count occurrences of each token. The vocab size is up to ~50257
    counts = torch.bincount(tokens)
    
    # Filter out tokens that never appear in our dataset
    non_zero_indices = torch.nonzero(counts).squeeze()
    
    # Handle the edge case of 0 or 1 unique token
    if non_zero_indices.dim() == 0:
        non_zero_indices = non_zero_indices.unsqueeze(0)
        
    non_zero_counts = counts[non_zero_indices]
    
    # Sort the tokens by frequency (descending order)
    sorted_counts, sorted_indices = torch.sort(non_zero_counts, descending=True)
    sorted_tokens = non_zero_indices[sorted_indices]
    
    # Get the GPT-2 tokenizer for decoding
    enc = tiktoken.get_encoding("gpt2")
    output_filename = "token_frequencies.txt"
    
    print(f"Writing frequencies to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"{'Token ID':<10} | {'Count':<10} | {'Decoded Value'}\n")
        f.write("-" * 60 + "\n")
        
        # Iterate over sorted tokens and write to file
        for token_id, count in zip(sorted_tokens.tolist(), sorted_counts.tolist()):
            try:
                # Decode the token ID back to text
                decoded = enc.decode([token_id])
                # Escape newlines and carriage returns so the output remains on one line
                decoded_repr = decoded.replace('\n', '\\n').replace('\r', '\\r')
            except Exception:
                decoded_repr = "<could not decode>"
                
            f.write(f"{token_id:<10} | {count:<10} | '{decoded_repr}'\n")

    print(f"Successfully analyzed dataset. Data saved to {output_filename}")

if __name__ == "__main__":
    analyze_dataset()
