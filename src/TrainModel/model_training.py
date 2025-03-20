from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

def train_model(X_train_enc, X_test_enc, y_train, y_test, num_epochs=5):
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Convert labels to long tensor format
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # Prepare datasets
    train_dataset = TensorDataset(
        X_train_enc['input_ids'], 
        X_train_enc['attention_mask'], 
        y_train_tensor
    )
    test_dataset = TensorDataset(
        X_test_enc['input_ids'], 
        X_test_enc['attention_mask'], 
        y_test_tensor
    )

    # Create DataLoader for training and testing
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)

    # Load pre-trained BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()  # CrossEntropyLoss for classification tasks

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)  # Calculate the loss
            loss.backward()
            optimizer.step()

        # Evaluate the model on test data
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch in test_dataloader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.logits, 1)  # Get the predicted class
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = correct / total
            print(f'Epoch {epoch + 1}/{num_epochs}, Accuracy: {accuracy:.4f}')
