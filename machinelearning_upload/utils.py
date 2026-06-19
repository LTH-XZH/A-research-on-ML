import matplotlib.pyplot as plt
import torch

# ---------------- accuracy评估 ----------------
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


# ---------------- loss画图 ----------------
def plot_loss(loss_list, name):
    plt.figure()
    plt.plot(loss_list)
    plt.title(name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(f"results/{name}.png")
    plt.close()