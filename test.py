import clip
import sys, os
from utils import *
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

# First argument should be model config `python test.py configs/base.yaml`
if __name__ == "__main__":
    # Config that should only change once
    DATA_PATH = "/Users/vikram/Documents/CMU/Multimodal/MultimodalRecipeProject/data/"


    # Load config
    if len(sys.argv) != 2:
        print("Usage: python test.py <config_path>")
        exit()

    config = OmegaConf.load(sys.argv[1])

    df = pd.read_csv(DATA_PATH  + "preprocessed_data/data.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get lastest checkpoint
    last_model = sorted(os.listdir(DATA_PATH + "checkpoints/" + config.exp_name))[-1]
    model_path = DATA_PATH + "checkpoints/" + config.exp_name + "/" + last_model
    print("Loading from '" + model_path + "'")
    
    model, preprocess = clip.load(model_path, device = device)
    model.eval()

    test_dataset = RecipeDataset(df, split = "test", device = device, data_path = DATA_PATH, context_length = config.context_length, image_preprocessor = preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = True)

    loss = 0.0
    
    criteron = torch.nn.CrossEntropyLoss()
    
    labels = torch.arange(config.batch_size).long().to(device)

    for image, text in tqdm(test_dataloader):
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        logits_per_image, logits_per_text = model(image, text)

        image_loss = criteron(logits_per_image, labels)
        text_loss = criteron(logits_per_text, labels)

        loss = (image_loss + text_loss)/2

        loss += torch.nn.functional.cosine_similarity(image_features, text_features).mean().item()

    print("Test loss:", loss/len(test_dataloader))