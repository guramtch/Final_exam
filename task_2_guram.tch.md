# Transformer Models and Cybersecurity Applications

Transformer models are sequence models that rely entirely on attention mechanisms instead of recurrence or convolutions. The central component is **multi‑head self‑attention**: for each token in a sequence, the model computes how strongly it should attend to every other token. This is done by projecting each token into query (Q), key (K), and value (V) vectors; attention weights are obtained from Q and K, and then used to form a weighted sum over the values. Multiple attention heads run in parallel, capturing different types of relationships such as “which host talked to which host” or “which verb is associated with which user”.

Because all tokens are processed in parallel, Transformers are highly efficient on modern hardware and can handle long sequences of events, which is valuable in security logs. Since the architecture itself does not encode order, the model adds **positional information** to token embeddings, typically through sinusoidal positional encodings or learned positional vectors. This lets the network distinguish “login, then privilege escalation” from “privilege escalation, then login”.

In cybersecurity, Transformers are useful wherever context matters:

- **Phishing email detection** – analysing the combination of subject, body, headers, and sender information.
- **Security log modelling** – predicting the next event or detecting anomalous sequences of log lines.
- **Threat report summarisation** – turning long, noisy incident reports into concise summaries for analysts.
- **Source‑code and script analysis** – finding suspicious patterns in PowerShell, JavaScript, or macros.

Compared with traditional bag‑of‑words classifiers or simple RNNs, Transformers can look far back in the sequence and focus attention on the most relevant events.

---

## Visualising Attention on a URL Request

Consider a short HTTP log line:

```text
"GET /login.php?user=admin 200"
```



import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Toy self-attention for an HTTP request
# --------------------------
tokens = ["GET", "/login.php", "?user=admin", "200"]

attention = np.array([
    [0.45, 0.30, 0.20, 0.05],  # GET
    [0.20, 0.40, 0.30, 0.10],  # /login.php
    [0.10, 0.35, 0.45, 0.10],  # ?user=admin
    [0.10, 0.25, 0.25, 0.40],  # 200
])

plt.figure()
plt.imshow(attention, interpolation="nearest")
plt.colorbar(label="attention weight")
plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
plt.yticks(range(len(tokens)), tokens)
plt.title("Toy attention over HTTP log tokens")
plt.tight_layout()
plt.savefig("task_2/attention.png", dpi=200)

# --------------------------
# 2. Sinusoidal positional encodings
# --------------------------
def sinusoidal_positional_encoding(max_len: int, d_model: int) -> np.ndarray:
    pe = np.zeros((max_len, d_model))
    position = np.arange(0, max_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

max_len = 30
d_model = 10
pe = sinusoidal_positional_encoding(max_len, d_model)

plt.figure()
for dim in range(5):  # plot first 5 dimensions
    plt.plot(pe[:max_len, dim], label=f"dim {dim}")
plt.xlabel("Position")
plt.ylabel("Encoding value")
plt.title("Sinusoidal positional encodings")
plt.legend()
plt.tight_layout()
plt.savefig("task_2/positional_encoding.png", dpi=200)

print("Saved attention.png and positional_encoding.png")

