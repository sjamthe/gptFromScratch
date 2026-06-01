import torch
import sys
import os

def main():
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), "rpn3_llm"))
    ckpt_path = "rpn3_llm/models/ut2.1M_2l_8h_384e_mlp4_phaseMask_True_cnt2_crd2_digitAbs_freezeCoordScale_sft_1-6_4num_BOS_300000.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    print("\nKeys in checkpoint:")
    for k in ckpt.keys():
        if k != 'model' and k != 'optimizer':
            print(f"  {k}: {type(ckpt[k])}")
            
    print("\nMetadata values:")
    for k in ['step', 'loss', 'val_loss', 'config', 'val_acc', 'val_perplexity']:
        if k in ckpt:
            if k == 'config':
                print(f"  config: {ckpt[k]}")
            else:
                print(f"  {k}: {ckpt[k]}")

if __name__ == "__main__":
    main()
