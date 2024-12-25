import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from clean_data import clean_data

# Очищение данных
original_file = 'youtube.csv'
cleaned_file = 'cleaned_data.csv'
clean_data(original_file, cleaned_file)

# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "albert-base-v2"  # Pre-trained ALBERT model
max_length = 128  # Максимальная длина токена
batch_size = 16  # Размер batch

tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # Assuming 3 categories
model.to(device)

data = pd.read_csv(cleaned_file)

# Валидация данных csv
required_columns = ["link", "description", "category", "cleaned_title"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"The CSV file must contain the following columns: {required_columns}")

# Маппинг категорий на integer для обучения модели
category_mapping = {label: idx for idx, label in enumerate(data["category"].unique())}
data["category"] = data["category"].map(category_mapping)
print(f"Category Mapping: {category_mapping}")

# Подготовка данных
texts = data["cleaned_title"].astype(str).tolist()
labels = data["category"].astype(int).tolist()

# Токенизирование
tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
tokens = {key: value.to(device) for key, value in tokens.items()}

# Обучение
predictions = []
with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_tokens = {key: value[i:i + batch_size] for key, value in tokens.items()}
        outputs = model(**batch_tokens)
        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()
        predictions.extend(batch_predictions)

# Вычисление необходимых метрик
precision = precision_score(labels, predictions, average="weighted", zero_division=1)
recall = recall_score(labels, predictions, average="weighted", zero_division=1)
f1 = f1_score(labels, predictions, average="weighted", zero_division=1)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")