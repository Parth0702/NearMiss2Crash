"""
Verify that saved checkpoints can be reloaded and used for inference.
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "experiments" / "distilbert_extension" / "models"
HFACS_CLASSES = [
    "Skill-Based Error",
    "Decision Error",
    "Violation",
    "Perceptual Error"
]

def test_checkpoint(model_name, checkpoint_path):
    """Load checkpoint and test inference on sample text."""
    print(f"\n{'='*60}")
    print(f"TESTING {model_name}")
    print(f"{'='*60}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print(f"Loaded: {checkpoint_path}")
    print(f"Config: {checkpoint['config']['BASE_CHECKPOINT']}, epochs={checkpoint['config']['EPOCHS']}")
    print(f"Test Metrics: Acc={checkpoint['test_metrics']['accuracy']:.4f}, Macro-F1={checkpoint['test_metrics']['macro_f1']:.4f}")

    # Recreate head
    head = nn.Linear(768, 4)
    head.load_state_dict({'weight': checkpoint['head_weight'], 'bias': checkpoint['head_bias']})
    head.eval()

    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    base_model = AutoModel.from_pretrained("distilbert-base-uncased")
    base_model.eval()

    # Test samples
    test_texts = [
        "The pilot failed to maintain proper airspeed during approach leading to stall",
        "Controller issued incorrect altitude clearance causing traffic conflict",
        "Pilot intentionally descended below minimum safe altitude",
        "First officer misread altimeter due to poor lighting in cockpit"
    ]

    print("\nInference on test samples:")
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            cls_rep = outputs.last_hidden_state[:, 0, :]
            logits = head(cls_rep)
            probs = torch.softmax(logits, dim=1).numpy()[0]
            pred_idx = probs.argmax()

            print(f"\n  Text: {text[:70]}...")
            for i, cls in enumerate(HFACS_CLASSES):
                print(f"    {cls}: {probs[i]:.4f}")
            print(f"  → Predicted: {HFACS_CLASSES[pred_idx]}")

    print(f"\n✓ {model_name} checkpoint verified successfully")
    return True

if __name__ == "__main__":
    test_checkpoint("Model_A_Crisp", MODELS_DIR / "model_a_crisp_head.pt")
    test_checkpoint("Model_B_Fuzzy", MODELS_DIR / "model_b_fuzzy_head.pt")
    print("\n" + "="*60)
    print("ALL CHECKPOINTS VERIFIED - READY FOR PHASE 4")
    print("="*60)