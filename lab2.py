import pandas as pd
import torch
from transformers import AlbertTokenizer, AlbertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from clean_data import clean_data

# Очищение данных
original_file = 'youtube.csv'
cleaned_file = 'cleaned_data.csv'
clean_data(original_file, cleaned_file)

data = pd.read_csv(cleaned_file)

# Валидация данных csv
required_columns = ["link", "description", "category", "cleaned_title"]
if not all(col in data.columns for col in required_columns):
    raise ValueError(f"CSV должен содержать колонки: {required_columns}")

# Маппинг категорий на integer для обучения модели
category_mapping = {label: idx for idx, label in enumerate(data["category"].unique())}
category_count = len(data["category"].unique())
data["category"] = data["category"].map(category_mapping)
print(f"Category Mapping: {category_mapping}")

# Конфигурация
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "albert-base-v2"  # Преобученная модель ALBERT
max_length = 128  # Максимальная длина токена
batch_size = 16  # Размер batch

tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=category_count)
model.to(device)

# Разделение данных на train/test
texts = data["cleaned_title"].astype(str).tolist()
labels = data["category"].astype(int).tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# Токенизирование
def tokenize_texts(texts, labels):
    tokens = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    labels = torch.tensor(labels).to(device)
    return tokens, labels

train_tokens, train_labels = tokenize_texts(train_texts, train_labels)
test_tokens, test_labels = tokenize_texts(test_texts, test_labels)

# Обучение
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

def train_model(model, train_tokens, train_labels, epochs=3):
    train_data = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_labels)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_fn = CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

train_model(model, train_tokens, train_labels, epochs=3)

# Оценка на тестовой выборке
def evaluate_model(model, test_tokens, test_labels):
    test_data = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_labels)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            batch_predictions = torch.argmax(logits, dim=-1).cpu().tolist()
            predictions.extend(batch_predictions)
            true_labels.extend(labels.cpu().tolist())

    precision = precision_score(true_labels, predictions, average="weighted", zero_division=1)
    recall = recall_score(true_labels, predictions, average="weighted", zero_division=1)
    f1 = f1_score(true_labels, predictions, average="weighted", zero_division=1)
    accuracy = accuracy_score(true_labels, predictions)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

evaluate_model(model, test_tokens, test_labels)
