import magicml as ml
import magicml.optim as optim
import magicml.nn as nn
from magicml.nn.functional import softmax, cross_entropy

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 0.0003
device = 'magic' if ml.is_magic_available() else 'stone'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

ml.set_seed(1337)

# Load and prepare the dataset
text = ml.load_text('input.txt')
chars = ml.unique_chars(text)
vocab_size = len(chars)
stoi, itos = ml.create_mappings(chars)
data = ml.tensor(ml.encode(text, stoi), dtype='long')
train_data, val_data = ml.split_data(data, split_ratio=0.9)


# Define the model
class MagicGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([nn.TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.init_weights()

    def init_weights(self):
        # Initialize weights magically
        self.apply(nn.magic_init)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(ml.range(T, device=device))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            probs = softmax(logits[:, -1, :], dim=-1)
            idx_next = ml.sample(probs, num_samples=1)
            idx = ml.concat([idx, idx_next], dim=1)
        return idx


# Instantiate the model and optimizer
model = MagicGPT(vocab_size, n_embd, n_head, n_layer, block_size, dropout).to(device)
optimizer = optim.MagicAdam(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        loss_train = ml.evaluate(model, train_data, batch_size, block_size, eval_iters, device)
        loss_val = ml.evaluate(model, val_data, batch_size, block_size, eval_iters, device)
        print(f"Step {iter}: Train Loss {loss_train:.4f}, Val Loss {loss_val:.4f}")

    xb, yb = ml.get_batch(train_data, batch_size, block_size, device)
    optimizer.zero_grad()
    logits, loss = model(xb, yb)
    loss.backward()
    optimizer.step()

# Generate text
context = ml.zeros((1, 1), dtype='long', device=device)
generated_text = model.generate(context, max_new_tokens=500)
print(ml.decode(generated_text[0].tolist(), itos))
