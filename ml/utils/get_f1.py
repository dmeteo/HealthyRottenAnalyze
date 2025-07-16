from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def get_class_f1(model, loader, id2label, device='cpu'):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    report = classification_report(
        all_targets, all_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True
    )
    f1_scores = [report[cl]['f1-score'] for cl in id2label.values()]
    supports = [report[cl]['support'] for cl in id2label.values()]
    df = pd.DataFrame({
        'Class': list(id2label.values()),
        'F1-score': f1_scores,
        'Support': supports
    })
    return df
