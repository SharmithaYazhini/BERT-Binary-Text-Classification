# Import necessary libraries
import pandas as pd
import torch
import sklearn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset




# read data
def read_data(file_path):
    return pd.read_csv(file_path)


# tokenize data
def tokenize_data(tokenizer, data):
    encodings = tokenizer(
        data["Terms"].tolist(),
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    return encodings

# train-test split
def perform_train_test_split(data, test_size=0.2, random_state=42):
    return train_test_split(data, test_size=test_size, random_state=random_state)

# preprocess data


def preprocess_data_for_training(data,tokenizer):

    input_texts, labels = data["Terms"].tolist(),data["Label"].tolist()
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
    labels = tokenizer(labels, return_tensors="pt", padding=True, truncation=True)
    return inputs, labels


# tensor datasets
def create_tensor_datasets(inputs, labels):
    return TensorDataset(inputs["input_ids"], labels["input_ids"])

# train the model

def train_model(model, train_loader, optimizer, num_epochs, test_loader, test_labels,scheduler):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        #scheduler.step()

        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).tolist()
                all_preds.extend(preds)

        accuracy = accuracy_score(test_labels, all_preds)
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy}")


def fine_tune_model(model, train_loader, optimizer, num_epochs, test_loader, test_labels,scheduler):
    best_accuracy = 0.0


    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).tolist()
                all_preds.extend(preds)
        accuracy = accuracy_score(test_labels, all_preds)
        print(f"Epoch {epoch + 1}, Accuracy: {accuracy}")

        #if epoch > 1 and accuracy <= best_accuracy:
        #    print("Early stopping.")
        #    break
        #else:
        #    best_accuracy = accuracy

# evaluate the model
def evaluate_model(model, test_loader, test_labels):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).tolist()
            all_preds.extend(preds)

    precision = precision_score(test_labels, all_preds)
    recall = recall_score(test_labels, all_preds)
    f1 = f1_score(test_labels, all_preds)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# save the model and tokenizer
def save_model_and_tokenizer(model, tokenizer, model_path, tokenizer_path):
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(tokenizer_path)

# make predictions 
def predict_new_data(model, tokenizer, input_text, label_mapping):
    input_encodings = tokenizer(
        input_text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**input_encodings)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    predicted_label = {v: k for k, v in label_mapping.items()}[predicted_class]
    print(f"Predicted Label: {predicted_label}")

    
class NLPDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

def main():
    
    data_path ="C:/Users/shanata/Downloads/NLP and Non NLP terms.csv"

    data = pd.read_csv(data_path)

    
    train_data, test_data = perform_train_test_split(data)

    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  

    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)


    label_mapping = {"NLP": 1, "Non-NLP": 0}

    train_labels = train_data["Label"].map(label_mapping).tolist()
    test_labels = test_data["Label"].map(label_mapping).tolist()

    train_encodings=tokenize_data(tokenizer,train_data)
    test_encodings=tokenize_data(tokenizer,test_data)

    train_dataset = NLPDataset(train_encodings, train_labels)
    test_dataset = NLPDataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

   


   
    num_epochs = 3

    
    train_model(model, train_loader, optimizer, num_epochs, test_loader, test_labels, scheduler)

    num_epochs= 6

    fine_tune_model(model, train_loader, optimizer, num_epochs, test_loader, test_labels, scheduler)

    
    evaluate_model(model, test_loader, test_labels)

    
    save_model_and_tokenizer(model, tokenizer, "model", "tokenizer")

    
    input_text = ["texting editor"]
    label_mapping = {"NLP": 1, "Non-NLP": 0}
    predict_new_data(model, tokenizer, input_text, label_mapping)

if __name__ == "__main__":


    main()
