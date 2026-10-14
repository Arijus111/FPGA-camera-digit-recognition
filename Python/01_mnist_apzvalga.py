# 01 – Susipažinimas su MNIST duomenimis
# Kiekvienas "# %%" yra atskiras langelis (cell). VS Code virš jo rodo "Run Cell".
# Paleisk juos po vieną iš eilės ir žiūrėk, ką kiekvienas išveda.

# %% Bibliotekos
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets

# %% MNIST užkrovimas
# Pirmą kartą parsiųs ~50 MB į aplanką python/data/
train = datasets.MNIST(root="data", train=True, download=True)
test = datasets.MNIST(root="data", train=False, download=True)

print("Treniravimo pavyzdžių:", len(train))
print("Testavimo pavyzdžių:", len(test))

# %% Vienas pavyzdys: kas jis toks?
img, label = train[0]          # paveiksliukas ir teisingas atsakymas
arr = np.array(img)            # paverčiam į paprastą skaičių masyvą

print("Teisingas atsakymas:", label)
print("Dydis:", arr.shape)                     # (28, 28)
print("Tipas:", arr.dtype)                     # uint8 = 8 bitai be ženklo
print("Min / max reikšmė:", arr.min(), arr.max())

# %% Tas pats paveiksliukas kaip skaičių lentelė
# Taip FPGA "matys" vaizdą: tiesiog 784 skaičiai 0..255
np.set_printoptions(linewidth=200)
print(arr)

# %% Keli pavyzdžiai su atsakymais
fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))
for i, ax in enumerate(axes.flat):
    img, label = train[i]
    ax.imshow(img, cmap="gray")
    ax.set_title(str(label))
    ax.axis("off")
plt.show()

# %% Kiek kokių skaitmenų yra?
counts = torch.bincount(train.targets, minlength=10)
for digit in range(10):
    print(f"Skaitmuo {digit}: {counts[digit].item()} pavyzdžių")

# %% Kur kadre paprastai būna skaitmuo?
# Suvidurkinam visus 60 000 paveiksliukų į vieną
mean_img = train.data.float().mean(dim=0)
plt.imshow(mean_img, cmap="gray")
plt.title("Vidutinis visų pavyzdžių vaizdas")
plt.colorbar()
plt.show()

# %% Ar kraštai aštrūs, ar yra pilkų pikselių?
data = train.data
nenuliniai = (data > 0).sum().item()
pilki = ((data > 0) & (data < 255)).sum().item()
print(f"Iš visų ne juodų pikselių {100 * pilki / nenuliniai:.1f} % yra pilki (ne 255)")
