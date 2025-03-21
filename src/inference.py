from transformers import BertTokenizer
import torch
from model_utils import load_model   
from data_processing import tokenize_data   
from model import create_model   


def make_inference(model_path, text_input):
    # Ensure the model is loaded correctly
    model = create_model(num_labels=2)  
    model = load_model(model, model_path)  

    # Check if a GPU or CPU is available for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the tokenizer (BERT tokenizer)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the input text (truncating and padding to fit BERT's input size)
    inputs = tokenizer(text_input, truncation=True, padding=True, max_length=512, return_tensors='pt')

    # Move the tokenized input to the device (GPU or CPU)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Move the model to the device (GPU or CPU)
    model = model.to(device)

    # Make predictions without tracking gradients (to save memory)
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask']) 
        predictions = torch.argmax(outputs.logits, dim=-1) 
         
    
    return predictions.cpu().numpy()
