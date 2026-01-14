import json
import matplotlib.pyplot as plt

# Update this path to your specific output folder
log_path = "models/qwen-recipe-ia3/trainer_state.json"

with open(log_path, 'r') as f:
    data = json.load(f)

# Extract data
train_steps = [x['step'] for x in data['log_history'] if 'loss' in x]
train_loss = [x['loss'] for x in data['log_history'] if 'loss' in x]

eval_steps = [x['step'] for x in data['log_history'] if 'eval_loss' in x]
eval_loss = [x['eval_loss'] for x in data['log_history'] if 'eval_loss' in x]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_loss, label='Training Loss', alpha=0.6)
plt.plot(eval_steps, eval_loss, label='Validation Loss', color='red', linewidth=2)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
