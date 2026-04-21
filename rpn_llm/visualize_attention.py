import os
import torch
import torch.nn.functional as F
import math
from model_rope import GPT, GPTConfig
from utils import RPNTokenizer

def generate_html_heatmap(matrix, labels, title, compact=False):
    """Generates an HTML/CSS heatmap table."""
    font_size = "8px" if compact else "10px"
    cell_size = "12px" if compact else "20px"
    
    html = f"<div style='margin-bottom: 20px;'>"
    html += f"<h4 style='margin: 5px 0;'>{title}</h4>"
    html += '<div style="display: inline-block; overflow-x: auto;">'
    html += f'<table style="border-collapse: collapse; font-family: monospace; font-size: {font_size}; border: 1px solid #ccc;">'
    
    # Header
    html += "<tr><th></th>"
    for label in labels:
        html += f'<th style="padding: 1px; transform: rotate(-90deg); height: 30px; min-width: {cell_size};">{label}</th>'
    html += "</tr>"
    
    for i, row in enumerate(matrix):
        html += f"<tr><th style='padding: 1px; text-align: right; min-width: 20px;'>{labels[i]}</th>"
        for val in row:
            # Power scaling to make high attention stand out more
            intensity = int(pow(val, 0.7) * 255) 
            bg_color = f"rgb(255, {255-intensity}, {255-intensity})" 
            html += f'<td style="background-color: {bg_color}; width: {cell_size}; height: {cell_size}; border: 1px solid #eee;" title="{val:.4f}"></td>'
        html += "</tr>"
    
    html += "</table></div></div>"
    return html

def visualize_attention(checkpoint_path, prompt_str, output_path="rpn_llm/inspection/report.html"):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 1. Load Model and Tokenizer
    print(f"Loading checkpoint {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    tokenizer = RPNTokenizer("rpn_llm/rpn-tokenizer.json")
    
    # 2. Prepare Input
    tokens = tokenizer.encode(prompt_str)
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    token_labels = [tokenizer.decode([t]) for t in tokens]
    # Clean labels for display
    token_labels = [t.replace(" ", "_") if t.strip() == "" else t for t in token_labels]
    
    # 3. Forward Pass with Attention Capture
    with torch.no_grad():
        logits, _, _, all_weights = model(x, return_attention=True)
        
    html_out = "<html><head><title>RPN Attention Report</title>"
    html_out += "<style>body { font-family: sans-serif; padding: 20px; background: #f5f5f5; } "
    html_out += ".layer-box { background: white; padding: 20px; margin-bottom: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); } "
    html_out += ".head-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; } </style></head><body>"
    
    html_out += f"<h1>RPN Mechanistic Analysis</h1>"
    html_out += f"<p><b>Prompt:</b> <code style='background: #eee; padding: 5px;'>{prompt_str}</code></p>"
    
    # --- LOGIT LENS ---
    html_out += "<div class='layer-box'><h2>Logit Lens (Internal State Predictions)</h2>"
    html_out += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%; background: white;'>"
    html_out += "<tr><th>Pass / Layer</th><th>Top 3 Guesses for Next Token</th></tr>"
    
    h = x
    h = model.transformer.wte(h)
    freq_cis = model.freqs_cis
    
    # Universal vs Sequential loop
    num_passes = config.n_layer
    for i in range(num_passes):
        if config.universal:
            # Universal mode: add pass embedding and use shared block
            h = h + model.pass_emb[i].view(1, 1, -1)
            h, _, _ = model.transformer.h(h, freq_cis, return_attention=False)
        else:
            # Sequential mode
            h, _, _ = model.transformer.h[i](h, freq_cis, return_attention=False)
            
        temp_logits = model.lm_head(model.transformer.ln_f(h))
        last_tok = temp_logits[0, -1, :]
        probs = F.softmax(last_tok, dim=-1)
        top_k = torch.topk(probs, 3)
        
        guesses = []
        for val, idx in zip(top_k.values, top_k.indices):
            token = tokenizer.decode([idx.item()])
            guesses.append(f"<b style='color: #d32f2f'>'{token}'</b> ({val.item()*100:.1f}%)")
        
        label = f"Pass {i+1}" if config.universal else f"Layer {i+1}"
        html_out += f"<tr><td>{label}</td><td>{' | '.join(guesses)}</td></tr>"
    html_out += "</table></div>"
    
    # --- ATTENTION MAPS ---
    for i in range(len(all_weights)):
        html_out += f"<div class='layer-box'><h2>Layer {i+1} Attention Mechanisms</h2>"
        
        # Mean Attention
        attn_mean = all_weights[i][0].mean(dim=0).cpu().numpy().tolist()
        html_out += generate_html_heatmap(attn_mean, token_labels, "Layer Mean Consensus (Global Flow)")
        
        # Individual Heads
        html_out += "<div class='head-grid'>"
        layer_weights = all_weights[i][0] # (nh, T, T)
        for head_idx in range(config.n_head):
            head_matrix = layer_weights[head_idx].cpu().numpy().tolist()
            html_out += generate_html_heatmap(head_matrix, token_labels, f"Head {head_idx+1}", compact=True)
        html_out += "</div>" # end head-grid
        html_out += "</div>" # end layer-box
        
    html_out += "</body></html>"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_out)
    
    print(f"\nComprehensive report generated at: {output_path}")

if __name__ == "__main__":
    import sys
    ckpt = sys.argv[2] if len(sys.argv) > 2 else "rpn_llm/models/UT3M_1-22_tens_comp_bracketed_final.pt"
    prompt = sys.argv[1] if len(sys.argv) > 1 else "(123)(456)+?<(32" # Example
    visualize_attention(ckpt, prompt)
