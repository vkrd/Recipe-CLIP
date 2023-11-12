import torch
import clip
from PIL import Image
from torch.utils.data import DataLoader
from utils import *
from omegaconf import OmegaConf
from tqdm import tqdm
import sys, os

if __name__ == "__main__":
    # Config that should only change once
    DATA_PATH = "/Users/vikram/Documents/CMU/Multimodal/MultimodalRecipeProject/data/"
    USE_WANDB = True
    DEBUG = True


    # Load config
    if len(sys.argv) != 2:
        print("Usage: python test.py <config_path>")
        exit()

    config = OmegaConf.load(sys.argv[1])

    if USE_WANDB:
        import wandb
        wandb.init(project="multimodal-CLIP", name=config.exp_name)
        wandb.config.update(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and freeze appropriate layers
    model, preprocess = clip.load(config.model_name, device = device)
    model.set_increased_context_length(config.context_length)

    for name, param in model.named_parameters():
        if name in ['text_projection', 'logit_scale']:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Load data
    df = pd.read_csv(DATA_PATH  + "preprocessed_data/data.csv")

    train_split = "val" if DEBUG else "train"
    train_dataset = RecipeDataset(df, split = train_split, device = device, data_path = DATA_PATH, context_length = config.context_length, image_preprocessor = preprocess)
    train_dataloader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True)

    val_dataset = RecipeDataset(df, split = "val", device = device, data_path = DATA_PATH, context_length = config.context_length, image_preprocessor = preprocess)
    val_dataloader = DataLoader(val_dataset, batch_size = config.batch_size, shuffle = True)

    # Start training
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    criteron = torch.nn.CrossEntropyLoss()

    labels = torch.arange(config.batch_size).long().to(device)

    for epoch in range(config.num_epochs):
        print("Starting epoch", epoch + 1)
        running_loss = 0.0
        model.train()
        for image, text in tqdm(train_dataloader):
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            
            image_loss = criteron(logits_per_image, labels)
            text_loss = criteron(logits_per_text, labels)

            loss = (image_loss + text_loss)/2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if DEBUG:
                break
        
        print("Evaluating on validation set")
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for image, text in tqdm(val_dataloader):
                image_features = model.encode_image(image)
                text_features = model.encode_text(text)

                logits_per_image, logits_per_text = model(image, text)
                
                image_loss = criteron(logits_per_image, labels)
                text_loss = criteron(logits_per_text, labels)

                loss = (image_loss + text_loss)/2
                val_loss += loss.item()
                if DEBUG:
                    break


        print("Train Loss:", running_loss/len(train_dataloader))
        print("Val Loss:", val_loss/len(val_dataloader))

        if USE_WANDB:
            wandb.log({
                "train_loss": running_loss/len(train_dataloader),
                "val_loss": val_loss/len(val_dataloader)
            })

        # Save model
        os.makedirs(DATA_PATH + "checkpoints", exist_ok = True)
        os.makedirs(DATA_PATH + "checkpoints/" + config.exp_name, exist_ok = True)
        torch.save(model.state_dict(), DATA_PATH + "checkpoints/" + config.exp_name + "/epoch_" + str(epoch) + ".pth")
        # torch.jit.save(torch.jit.script(model), DATA_PATH + "checkpoints/" + config.exp_name + "/epoch_" + str(epoch) + ".pt")
        