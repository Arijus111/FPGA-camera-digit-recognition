# 03 – Kvantavimas: float modelis -> sveikieji skaičiai, kuriuos supras FPGA
# Paleisk langelius po vieną iš eilės. Reikia mlp_float.pt iš 02 failo.
#
# Idėja: FPGA nemoka (efektyviai) skaičiuoti su trupmenomis.
# Todėl kiekvieną realų skaičių užrašom kaip   tikra_reikšmė ≈ sveikas_skaičius × mastelis
# Sveikus skaičius saugom BRAM'e, o masteliai lieka čia, Python'e.
# Gale gausim:
#   - svorius int8 (-127..127)  -> weights/*.mem
#   - poslinkius int32          -> weights/*.mem
#   - vieną "perskaliavimo" daugiklį ir poslinkį bitais -> weights/params.vh
#   - golden modelį: funkciją, kuri skaičiuoja BIT Į BITĄ taip, kaip skaičiuos FPGA
#   - testinius vektorius simuliacijai -> vectors/

# %% Bibliotekos
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import datasets
import matplotlib.pyplot as plt

WEIGHTS_DIR = Path("../weights")
VECTORS_DIR = Path("../vectors")
WEIGHTS_DIR.mkdir(exist_ok=True)
VECTORS_DIR.mkdir(exist_ok=True)

# %% Užkraunam ištreniruotą modelį ir duomenis
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
model.load_state_dict(torch.load("mlp_float.pt"))

# Svoriai kaip paprasti numpy masyvai
W1 = model[1].weight.detach().numpy().astype(np.float64)   # 64 x 784
b1 = model[1].bias.detach().numpy().astype(np.float64)     # 64
W2 = model[3].weight.detach().numpy().astype(np.float64)   # 10 x 64
b2 = model[3].bias.detach().numpy().astype(np.float64)     # 10

# Pikseliai TOKIE, kokius gaus FPGA: sveiki skaičiai 0..255, be jokio dalijimo
train = datasets.MNIST("data", train=True, download=True)
test = datasets.MNIST("data", train=False, download=True)
X_train = train.data.reshape(-1, 784).numpy().astype(np.int64)
X_test = test.data.reshape(-1, 784).numpy().astype(np.int64)
y_test = test.targets.numpy()

print("Svorių dydžiai:", W1.shape, b1.shape, W2.shape, b2.shape)
print("Testo vaizdų:", X_test.shape)

# %% Atskaitos taškas: float modelis, perrašytas numpy
# Tai ta pati formulė kaip PyTorch, tik "ranka" - kad matytum, kad nieko magiško nėra.
def float_modelis(X):
    x = X / 255.0                          # tas pats, ką darė ToTensor()
    h = np.maximum(0, x @ W1.T + b1)       # 1 sluoksnis + ReLU
    out = h @ W2.T + b2                    # 2 sluoksnis
    return out, h

out_f, h_f = float_modelis(X_test)
pred_f = out_f.argmax(axis=1)
print(f"Float modelio tikslumas: {100 * (pred_f == y_test).mean():.2f} %")

# %% 1 žingsnis: svoriai -> int8
# Surandam didžiausią |svorį| ir jį priskiriam 127. Visi kiti - proporcingai.
#   mastelis = max|W| / 127
#   W_int    = round(W / mastelis)
def kvantuoti_int8(W):
    mastelis = np.abs(W).max() / 127
    W_int = np.clip(np.round(W / mastelis), -127, 127).astype(np.int64)
    return W_int, mastelis

W1q, s1 = kvantuoti_int8(W1)
W2q, s2 = kvantuoti_int8(W2)

print(f"1 sluoksnio mastelis s1 = {s1:.6f}  (vienas int žingsnis = {s1:.6f} realaus svorio)")
print(f"2 sluoksnio mastelis s2 = {s2:.6f}")
print()
print("Pavyzdys - pirmi 6 neurono 0 svoriai prie centrinių pikselių:")
idx = slice(406, 412)
print("  float:          ", np.round(W1[0, idx], 4))
print("  int8:           ", W1q[0, idx])
print("  int8 × mastelis:", np.round(W1q[0, idx] * s1, 4))
print(f"Didžiausia apvalinimo paklaida: {np.abs(W1q * s1 - W1).max():.6f}  (≤ s1/2)")

# %% 2 žingsnis: paslėptojo sluoksnio mastelis (kalibravimas)
# Po 1 sluoksnio ir ReLU gaunam 64 teigiamus skaičius. FPGA juos laikys 8 bitais be ženklo (0..255).
# Reikia žinoti, kokio dydžio jie būna - tam paleidžiam float modelį per 10 000 treniravimo vaizdų.
_, h_kalibr = float_modelis(X_train[:10000])
h_max = h_kalibr.max()

sx = 1 / 255          # įvesties mastelis: pikselis 255 atitinka 1.0
sh = h_max / 255      # paslėptojo sluoksnio mastelis: didžiausia reikšmė atitinka 255

print(f"Didžiausia paslėptojo sluoksnio reikšmė: {h_max:.3f}")
print(f"Paslėptojo sluoksnio mastelis sh = {sh:.6f}")

plt.hist(h_kalibr.flatten(), bins=100, log=True)
plt.title("Paslėptojo sluoksnio reikšmės po ReLU (log skalė)")
plt.show()

# %% 3 žingsnis: poslinkiai -> int32
# Kaupiklyje (akumuliatoriuje) sudedam pikselis_int × svoris_int.
# Tokios sandaugos mastelis yra sx × s1, todėl poslinkį reikia išreikšti tuo pačiu masteliu.
b1q = np.round(b1 / (sx * s1)).astype(np.int64)
b2q = np.round(b2 / (sh * s2)).astype(np.int64)

print("b1q pavyzdžiai:", b1q[:8])
print("b2q:", b2q)

# %% 4 žingsnis: perskaliavimas tarp sluoksnių
# Po 1 sluoksnio kaupiklio vertė turi mastelį sx·s1, o 2 sluoksniui reikia mastelio sh.
# Todėl ją reikia padauginti iš M = sx·s1/sh. M yra mažas trupmeninis skaičius.
# FPGA daro taip:  M ≈ m / 2^n   ->   rezultatas = (vertė × m) >> n
# Tai viena daugyba (DSP) ir poslinkis bitais (tik laidai).
M = sx * s1 / sh

n = 0
while M * 2 ** (n + 1) < 2 ** 15:     # parenkam n taip, kad m tilptų į 15 bitų
    n += 1
m = int(round(M * 2 ** n))

print(f"Tikrasis daugiklis M  = {M:.10f}")
print(f"FPGA daugiklis m      = {m}  (15 bitų)")
print(f"Poslinkis n           = {n}")
print(f"m / 2^n               = {m / 2 ** n:.10f}")
print(f"Santykinė paklaida    = {abs(m / 2 ** n - M) / M:.2e}")

# %% GOLDEN MODELIS: tik sveikieji skaičiai, lygiai kaip FPGA
# Kiekviena eilutė čia = vienas aparatūros blokas. Verilog rezultatai turės sutapti BIT Į BITĄ.
def int_modelis(X):
    # 1 sluoksnis: MAC per 784 pikselius + poslinkis
    acc1 = X @ W1q.T + b1q                          # signed kaupiklis

    # ReLU: jei neigiama -> 0
    relu1 = np.maximum(acc1, 0)

    # Perskaliavimas: (x * m + apvalinimas) >> n, tada apribojam iki 255
    h = (relu1 * m + (1 << (n - 1))) >> n
    h = np.minimum(h, 255)                          # 8 bitai be ženklo

    # 2 sluoksnis: MAC per 64 neuronus + poslinkis
    acc2 = h @ W2q.T + b2q                          # 10 rezultatų

    return acc2, h, acc1

out_q, h_q, acc1_q = int_modelis(X_test)
pred_q = out_q.argmax(axis=1)    # FPGA: pirmas didžiausias (kaip ir np.argmax)

print(f"Float modelis:  {100 * (pred_f == y_test).mean():.2f} %")
print(f"Int modelis:    {100 * (pred_q == y_test).mean():.2f} %")
print(f"Atsakymai sutampa {100 * (pred_q == pred_f).mean():.2f} % atvejų")

# %% Kiek bitų reikės aparatūroje?
def bitu_su_zenklu(x):
    return int(np.ceil(np.log2(np.abs(x).max() + 1))) + 1

sandauga = (np.maximum(acc1_q, 0) * m)
print("Signalų pločiai (su ženklo bitu), išmatuoti per visą testą:")
print(f"  pikselis × svoris1       : 8 bit (be ženklo) × 8 bit -> {bitu_su_zenklu(255 * 127)} bit")
print(f"  1 sluoksnio kaupiklis     : {bitu_su_zenklu(acc1_q)} bit")
print(f"  poslinkiai b1             : {bitu_su_zenklu(b1q)} bit")
print(f"  ReLU × m (prieš >> n)     : {bitu_su_zenklu(sandauga)} bit")
print(f"  paslėptas sluoksnis h     : 8 bit be ženklo (0..255)")
print(f"  2 sluoksnio kaupiklis     : {bitu_su_zenklu(out_q)} bit")
print(f"  poslinkiai b2             : {bitu_su_zenklu(b2q)} bit")
print()
print(f"Teorinis blogiausias 1 sl. kaupiklis: 784 × 255 × 127 = {784 * 255 * 127}"
      f" -> {bitu_su_zenklu(784 * 255 * 127)} bit")
print("Saugus pasirinkimas FPGA: 32 bitų kaupiklis abiem sluoksniams.")

# %% Eksportas į .mem failus ($readmemh formatas)
# Kiekvienoje eilutėje viena reikšmė šešioliktainiu (hex) formatu.
# Neigiami skaičiai užrašomi papildomuoju kodu (two's complement), pvz. -1 -> FF.
def rasyti_mem(failas, reiksmes, bitai):
    kauke = (1 << bitai) - 1
    plotis = bitai // 4
    with open(failas, "w") as f:
        for v in np.asarray(reiksmes).flatten():
            f.write(f"{int(v) & kauke:0{plotis}X}\n")
    print(f"  {failas}  ({np.asarray(reiksmes).size} eilučių, {bitai} bit)")

print("Rašomi failai:")
# Adresas = neuronas * 784 + pikselis   (eilutė po eilutės)
rasyti_mem(WEIGHTS_DIR / "w1.mem", W1q, 8)
rasyti_mem(WEIGHTS_DIR / "b1.mem", b1q, 32)
# Adresas = išėjimas * 64 + neuronas
rasyti_mem(WEIGHTS_DIR / "w2.mem", W2q, 8)
rasyti_mem(WEIGHTS_DIR / "b2.mem", b2q, 32)

with open(WEIGHTS_DIR / "params.vh", "w") as f:
    f.write("// Sugeneruota 03_kvantavimas.py - neredaguoti ranka\n")
    f.write(f"localparam integer REQ_MULT  = {m};  // perskaliavimo daugiklis\n")
    f.write(f"localparam integer REQ_SHIFT = {n};  // poslinkis bitais\n")
print(f"  {WEIGHTS_DIR / 'params.vh'}  (REQ_MULT={m}, REQ_SHIFT={n})")

# %% Testiniai vektoriai simuliacijai
# Paimam N testo vaizdų ir išsaugom: įvestį, paslėptąjį sluoksnį, 10 išėjimų ir atsakymą.
# Testbench'e palyginsi savo modulio rezultatus su šiais failais.
N = 20
X_vec = X_test[:N]
out_vec, h_vec, acc1_vec = int_modelis(X_vec)

print("Rašomi failai:")
rasyti_mem(VECTORS_DIR / "input.mem", X_vec, 8)        # vaizdas k: eilutės k*784 .. k*784+783
rasyti_mem(VECTORS_DIR / "acc1.mem", acc1_vec, 32)     # vaizdas k: eilutės k*64 ..  (prieš ReLU)
rasyti_mem(VECTORS_DIR / "hidden.mem", h_vec, 8)       # vaizdas k: eilutės k*64 ..  (po perskaliavimo)
rasyti_mem(VECTORS_DIR / "scores.mem", out_vec, 32)    # vaizdas k: eilutės k*10 ..
rasyti_mem(VECTORS_DIR / "expected.mem", out_vec.argmax(axis=1), 4)

with open(VECTORS_DIR / "README.txt", "w") as f:
    f.write(f"{N} testo vaizdų iš MNIST.\n")
    f.write("expected.mem - ką turi atsakyti FPGA (golden modelio argmax)\n")
    f.write("Tikri atsakymai (žmogaus etiketės): " + " ".join(map(str, y_test[:N])) + "\n")

# %% Peržiūra: vaizdas, float ir int atsakymai
fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
for k, ax in enumerate(axes.flat):
    ax.imshow(X_vec[k].reshape(28, 28), cmap="gray")
    ax.set_title(f"float {pred_f[k]} | int {out_vec[k].argmax()}", fontsize=8)
    ax.axis("off")
plt.show()

print("Pirmo vaizdo 10 išėjimų (int):", out_vec[0])
print("Didžiausias:", out_vec[0].argmax(), " tikras atsakymas:", y_test[0])
