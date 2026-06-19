import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from models import get_cnn, get_swin
from utils import evaluate, plot_loss

# ---------------- device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- dataset ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

trainset = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

testset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


# ---------------- train function ----------------
def train(model, name, epochs=3):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    loss_list = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss) or torch.isinf(loss):
                print("NaN/Inf detected, skip batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        loss_list.append(avg_loss)

        acc = evaluate(model, testloader, device)

        print(f"{name} Epoch {epoch+1} | loss={avg_loss:.4f} | acc={acc:.4f}")

    plot_loss(loss_list, name)
    return acc


# ---------------- main ----------------
if __name__ == "__main__":

    print("Loading models...")

    cnn = get_cnn()
    swin = get_swin()

    print("\n===== CNN =====")
    cnn_acc = train(cnn, "CNN")

    print("\n===== Swin =====")
    swin_acc = train(swin, "SwinV2")

    print("\n===== FINAL RESULT =====")
    print(f"CNN   : {cnn_acc:.4f}")
    print(f"Swin: {swin_acc:.4f}")