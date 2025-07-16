import time

def measure_inference_time(model, loader, device='cpu', repeats=3):
    model.eval()
    times = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            # Warmup
            for _ in range(repeats):
                _ = model(images)
            start = time.time()
            _ = model(images)
            elapsed = time.time() - start
            times.append(elapsed / len(images))  # sec per image
            break  # достаточно одной batch
    mean_time = np.mean(times)
    return mean_time