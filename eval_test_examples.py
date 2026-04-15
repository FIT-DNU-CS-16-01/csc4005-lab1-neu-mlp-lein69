import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
from src.dataset import create_dataloaders, FlattenTransform
from src.model import MLPClassifier

def load_best_model(run_name='baseline_adamw', img_size=64):
    """Load best model from outputs"""
    model_path = Path(f'outputs/{run_name}/best_model.pt')
    metrics_path = Path(f'outputs/{run_name}/metrics.json')
    
    # Load metrics to get class_names
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    class_names = metrics['class_names']
    num_classes = len(class_names)
    input_dim = img_size * img_size  # Flattened image
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MLPClassifier(input_dim=input_dim, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, class_names, device

def get_predictions_on_test_set(model, class_names, device, data_dir='NEU-CLS_extracted', img_size=64):
    """Get predictions on test set and separate correct vs incorrect"""
    # Load test set
    split_data = create_dataloaders(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=1,  # Process one at a time to get individual predictions
        augment=False
    )
    
    test_loader = split_data.test_loader
    correct_preds = []
    incorrect_preds = []
    
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            pred_labels = logits.argmax(dim=1)
            confidence = probs.max(dim=1)[0]
        
        true_label = labels.item()
        pred_label = pred_labels.item()
        conf_score = confidence.item()
        
        pred_data = {
            'true_label': class_names[true_label],
            'pred_label': class_names[pred_label],
            'confidence': conf_score,
            'true_idx': true_label,
            'pred_idx': pred_label
        }
        
        if true_label == pred_label:
            correct_preds.append(pred_data)
        else:
            incorrect_preds.append(pred_data)
    
    return correct_preds, incorrect_preds

def visualize_examples(correct_preds, incorrect_preds, class_names, output_path='outputs/baseline_adamw/test_evaluation.txt'):
    """Save test evaluation summary"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TEST SET EVALUATION - BEST MODEL (baseline_adamw)\n")
        f.write("="*70 + "\n\n")
        
        total = len(correct_preds) + len(incorrect_preds)
        accuracy = len(correct_preds) / total if total > 0 else 0
        
        f.write(f"Total test samples: {total}\n")
        f.write(f"Correct predictions: {len(correct_preds)} ({accuracy*100:.2f}%)\n")
        f.write(f"Incorrect predictions: {len(incorrect_preds)} ({(1-accuracy)*100:.2f}%)\n\n")
        
        f.write("-"*70 + "\n")
        f.write("SAMPLE CORRECT PREDICTIONS (Top 4)\n")
        f.write("-"*70 + "\n")
        for i, pred in enumerate(correct_preds[:4]):
            f.write(f"{i+1}. {pred['true_label']:20s} -> Predicted: {pred['pred_label']:20s} ")
            f.write(f"(Confidence: {pred['confidence']:.2%})\n")
        
        f.write("\n" + "-"*70 + "\n")
        f.write("SAMPLE INCORRECT PREDICTIONS (Top 4)\n")
        f.write("-"*70 + "\n")
        for i, pred in enumerate(incorrect_preds[:4]):
            f.write(f"{i+1}. True: {pred['true_label']:20s} -> Predicted: {pred['pred_label']:20s} ")
            f.write(f"(Confidence: {pred['confidence']:.2%})\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"✓ Saved test evaluation to {output_path}")

def print_summary_stats(correct_preds, incorrect_preds, class_names):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("TEST SET EVALUATION SUMMARY")
    print("="*60)
    
    total = len(correct_preds) + len(incorrect_preds)
    accuracy = len(correct_preds) / total if total > 0 else 0
    
    print(f"\nTotal test samples: {total}")
    print(f"Correct predictions: {len(correct_preds)} ({accuracy*100:.2f}%)")
    print(f"Incorrect predictions: {len(incorrect_preds)} ({(1-accuracy)*100:.2f}%)")
    
    print("\n" + "-"*60)
    print("SAMPLE CORRECT PREDICTIONS:")
    print("-"*60)
    for i, pred in enumerate(correct_preds[:4]):
        print(f"{i+1}. {pred['true_label']:20s} → Predicted: {pred['pred_label']:20s} (Conf: {pred['confidence']:.2%})")
    
    print("\n" + "-"*60)
    print("SAMPLE INCORRECT PREDICTIONS:")
    print("-"*60)
    for i, pred in enumerate(incorrect_preds[:4]):
        print(f"{i+1}. {pred['true_label']:20s} → Predicted: {pred['pred_label']:20s} (Conf: {pred['confidence']:.2%})")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    # Load best model
    model, class_names, device = load_best_model('baseline_adamw')
    print(f"✓ Loaded model for {len(class_names)} classes")
    
    # Get predictions
    correct_preds, incorrect_preds = get_predictions_on_test_set(model, class_names, device)
    
    # Print summary
    print_summary_stats(correct_preds, incorrect_preds, class_names)
    
    # Visualize examples
    visualize_examples(correct_preds, incorrect_preds, class_names)
    
    print("\n✓ Test evaluation complete!")
