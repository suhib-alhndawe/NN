import torch

def save_model(model, model_path='model.pth'):
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

def load_model(model, model_path='model.pth'):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
