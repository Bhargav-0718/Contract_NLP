import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from LegalBERT import id2label  # <-- your original label mapping

# Load the fine-tuned model
model_path = "legalbert_cuad_paragraph"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Put model in evaluation mode
model.eval()

# Example test input (clause)
test_clause = "Either party may terminate this Agreement upon thirty (30) day."

# Tokenize
inputs = tokenizer(test_clause, return_tensors="pt", truncation=True, padding=True, max_length=512)

# Run prediction
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()


# ✅ Use your custom mapping
predicted_label = id2label[predicted_class_id]

print(f"Test Clause: {test_clause}")
print(f"Predicted Label: {predicted_label}")
