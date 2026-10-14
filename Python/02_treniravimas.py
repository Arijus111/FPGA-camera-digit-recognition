# 02 – Modelio (784 -> 64 -> 10) treniravimas
# Paleisk langelius po vieną iš eilės.

# %% Bibliotekos
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

torch.manual_seed(0)   # kad kiekvieną kartą gautum tą patį rezultatą

# %% Duomenys
# ToTensor() pikselius 0..255 paverčia į 0.0..1.0 (tiesiog padalija iš 255).
# Jokio papildomo normalizavimo - kad FPGA įvestis liktų paprasti 8 bitų pikseliai.
transform = transforms.ToTensor()
train = datasets.MNIST("data", train=True, download=True, transform=transform)
test = datasets.MNIST("data", train=False, download=True, transform=transform)

# DataLoader paduoda duomenis porcijomis (batch) po 64 paveiksliukus
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=1000, shuffle=False)

# %% Modelis - lygiai ta formulė, kurią darysi FPGA
model = nn.Sequential(
    nn.Flatten(),          # 28x28 -> 784 skaičių eilutė
    nn.Linear(784, 64),    # 64 neuronai: kiekvienas = suma(pikselis * svoris) + poslinkis
    nn.ReLU(),             # neigiami -> 0
    nn.Linear(64, 10),     # 10 išėjimų, po vieną kiekvienam skaitmeniui
)
print(model)

parametru = sum(p.numel() for p in model.parameters())
print("Parametrų (svorių + poslinkių) iš viso:", parametru)
# Patikrink ranka: 784*64 + 64 + 64*10 + 10 = ?

# %% Pagalbinė funkcija: koks procentas testo atspėtas teisingai
def tikslumas(model, loader):
    model.eval()
    teisingi = 0
    with torch.no_grad():                        # tik skaičiuojam, nesimokom
        for x, y in loader:
            spejimas = model(x).argmax(dim=1)    # didžiausio išėjimo numeris = atsakymas
            teisingi += (spejimas == y).sum().item()
    return teisingi / len(loader.dataset)

print(f"Tikslumas PRIEŠ treniravimą: {100 * tikslumas(model, test_loader):.1f} %")
# Turėtų būti ~10 % - t.y. atsitiktinis spėliojimas

# %% Treniravimas
loss_fn = nn.CrossEntropyLoss()                            # matuoja, kiek modelis klysta
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # koreguoja svorius

for epocha in range(8):          # 8 kartus pereinam per visus 60 000 pavyzdžių
    model.train()
    bendra_klaida = 0.0
    for x, y in train_loader:
        isejimai = model(x)              # 1. paskaičiuojam formulę
        klaida = loss_fn(isejimai, y)    # 2. kiek suklydom
        optimizer.zero_grad()
        klaida.backward()                # 3. kuria kryptimi keisti kiekvieną svorį
        optimizer.step()                 # 4. truputį pakeičiam svorius
        bendra_klaida += klaida.item() * x.size(0)

    print(f"Epocha {epocha + 1}: klaida = {bendra_klaida / len(train):.4f}, "
          f"tikslumas = {100 * tikslumas(model, test_loader):.2f} %")

# %% Išsaugom ištreniruotus svorius
torch.save(model.state_dict(), "mlp_float.pt")
print("Išsaugota: mlp_float.pt")

# %% Svorių diapazonai - prireiks kvantavimui
for vardas, p in model.named_parameters():
    print(f"{vardas:10s} dydis={tuple(p.shape)}  min={p.min().item():+.3f}  max={p.max().item():+.3f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 3))
for ax, (vardas, p) in zip(axes, model.named_parameters()):
    ax.hist(p.detach().flatten().numpy(), bins=100)
    ax.set_title(vardas)
plt.show()

# %% Ką "mato" pirmo sluoksnio neuronai?
# Kiekvieno neurono 784 svorius sudedam atgal į 28x28 paveiksliuką.
# Šviesu = tas pikselis didina neurono reikšmę, tamsu = mažina.
svoriai = model[1].weight.detach()       # dydis 64 x 784
fig, axes = plt.subplots(4, 16, figsize=(16, 4.5))
for i, ax in enumerate(axes.flat):
    ax.imshow(svoriai[i].reshape(28, 28), cmap="gray")
    ax.axis("off")
plt.suptitle("64 neuronų svoriai kaip paveiksliukai")
plt.show()

# %% Kur modelis klysta?
model.eval()
x, y = next(iter(test_loader))
with torch.no_grad():
    spejimas = model(x).argmax(dim=1)
klaidos = (spejimas != y).nonzero().flatten()[:16]

fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for ax, idx in zip(axes.flat, klaidos):
    ax.imshow(x[idx, 0], cmap="gray")
    ax.set_title(f"tikra {y[idx].item()}, spėjo {spejimas[idx].item()}", fontsize=8)
    ax.axis("off")
plt.show()
