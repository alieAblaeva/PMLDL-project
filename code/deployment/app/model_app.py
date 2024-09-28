from PIL import Image
import os
from torchvision import transforms
from age_pred_dropout03 import AgeDetectionCNN
import torch
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


model = AgeDetectionCNN()
model_path = os.path.join('age_pred_bs64_ne45_do03.pth')

model.load_state_dict(torch.load(model_path))


preprocess = transforms.Compose([
    transforms.Resize((200, 200)), 
    transforms.ToTensor(),  
])



def model_pipeline(image: Image):
    encoded = preprocess(image)
    img_tensor = encoded.unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
    print('output: ', output)
    return torch.squeeze(output).item()