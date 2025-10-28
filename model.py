import torch
from arch import ConvLSTM

def Load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(path, map_location=device)
    
    model1 = ConvLSTM().to(device)
    
    # Use 4 timesteps to match your actual input (not 5)
    dummy_input = torch.randn(1, 5, 23, 256, 256).to(device)  # Changed from 5 to 4
    _ = model1(dummy_input)
    
    model1.load_state_dict(state_dict)
    return model1, device

