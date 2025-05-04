import torch
from transformers import BertTokenizer, BertForSequenceClassification
from bitflip_attack.attacks.bit_flip_attack import BitFlipAttack
import copy
import traceback # Import traceback for error details

# --- Configuration ---
MODEL_NAME = "prajjwal1/bert-tiny" # Use a small standard model
NUM_LABELS = 2 # Example: Binary classification
MAX_BIT_FLIPS = 5 # How many bits the attack should flip
TARGET_CLASS = 1 # Example: Target the positive class

# --- Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}")
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(device)
model.eval() # Set to evaluation mode

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

# --- Create Simple Dummy Data ---
# Using just a few examples to test functionality
texts = [
    "This is a positive example.",
    "This is another positive example!",
    "Here is a negative sentence.",
    "This one is definitely negative.",
    "Maybe this leans positive?", # Ambiguous
    "This could be negative." # Ambiguous
]
# Corresponding labels (0 for negative, 1 for positive)
labels = [1, 1, 0, 0, 1, 0]

# Tokenize data
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)
true_labels = torch.tensor(labels).to(device)

# --- Simple Evaluation Function ---
def evaluate_simple(current_model, input_ids, attention_mask, true_labels):
    correct = 0
    total = 0
    with torch.no_grad():
        outputs = current_model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=1)
        total = true_labels.size(0)
        correct = (predictions == true_labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    print(f"  Correct: {correct}/{total}, Accuracy: {accuracy:.4f}")
    return accuracy

# --- Evaluation BEFORE Attack ---
print("\n--- Evaluating BEFORE attack ---")
initial_accuracy = evaluate_simple(model, input_ids, attention_mask, true_labels)

# --- Prepare for Attack ---
# The attack needs a dataset object (or something list-like with __len__ and __getitem__)
# Let's create a simple list-based dataset wrapper
class SimpleDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Return item suitable for the attack's internal evaluation
        # The attack's custom_forward_fn expects the batch format
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx].clone().detach() # Attack might need labels internally
        # Attack custom_forward_fn needs dict or tuple(dict, tensor)
        # Let's return the tuple format expected by some attack loops
        input_dict = {k: v for k, v in item.items() if k != 'labels'}
        label_tensor = item['labels']
        return input_dict, label_tensor # Return (inputs_dict, targets) tuple

    def __len__(self):
        return len(self.labels)

attack_dataset = SimpleDataset(inputs, true_labels)

# The attack also needs a custom forward function
# Use the same one from the main example script if it's compatible
# (Make sure it's defined above or imported)
def custom_forward_fn(model, batch):
    """Handles dict or tuple(dict, tensor) inputs"""
    input_dict = None
    if isinstance(batch, tuple) and len(batch) == 2:
        input_dict = batch[0]
    elif isinstance(batch, dict):
        input_dict = batch
    else:
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if isinstance(input_dict, dict) and 'input_ids' in input_dict and 'attention_mask' in input_dict:
        input_ids = input_dict['input_ids'].to(model.device)
        attention_mask = input_dict['attention_mask'].to(model.device)
        return model(input_ids=input_ids, attention_mask=attention_mask).logits
    else:
        raise ValueError("Could not extract valid input_dict")

# --- Run Standard Bit Flip Attack ---
print("\n--- Running Standard Bit Flip Attack ---")
# Copy model state so original isn't modified directly by attack object setup
attack_model = copy.deepcopy(model)
attack_model.eval()

try:
    standard_attack = BitFlipAttack(
        model=attack_model, # Pass the copied model
        dataset=attack_dataset,
        max_bit_flips=MAX_BIT_FLIPS,
        device=device,
        custom_forward_fn=custom_forward_fn
    )

    # Run the attack (using minimal GA parameters for speed)
    attack_results = standard_attack.perform_attack(
        target_class=TARGET_CLASS,
        num_candidates=10, # Fewer candidates
        population_size=5, # Smaller population
        generations=3      # Fewer generations
    )
    print("\nAttack finished. Results (internal):")
    print(attack_results)

    # --- Evaluation AFTER Attack ---
    print("\n--- Evaluating AFTER attack ---")
    # The attack modifies the 'attack_model' in place
    final_accuracy = evaluate_simple(attack_model, input_ids, attention_mask, true_labels)

    print("\n--- Summary ---")
    print(f"Initial Accuracy: {initial_accuracy:.4f}")
    print(f"Final Accuracy:   {final_accuracy:.4f}")
    print(f"Accuracy Drop:    {initial_accuracy - final_accuracy:.4f}")
    print(f"Bits Flipped by Attack: {attack_results.get('bits_flipped', 'N/A')}")

except Exception as e:
    print(f"ERROR during attack or evaluation: {e}")
    traceback.print_exc() 