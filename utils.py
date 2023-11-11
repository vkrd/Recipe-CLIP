from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from tqdm import tqdm
import clip

class RecipeDataset(Dataset):
    def __init__(self, data: pd.DataFrame, split: str, device: str, data_path: str, context_length: int = 512, image_preprocessor = None):
        self.split = split
        self.data = data[data.split == split]
        self.device = device
        self.prefix = data_path + "preprocessed_data/"
        self.processor = image_preprocessor

        print("Preprocessing Recipes")
        self.recipes = []
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            ingredients = row["ingredients"].split('/t')
            instructions = row["instructions"].split('/t')
            instructions_list = ""
            for idx, inst in enumerate(instructions):
                instructions_list += str(idx + 1) + ") " + inst + "\n"
            
            recipe = "\n".join([
                "Ingredients: ",
                "\n".join(ingredients),
                "\nInstructions: ",
                instructions_list
            ])

            self.recipes.append(clip.tokenize([recipe], context_length=context_length, truncate=True)[0])

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image = self.processor(Image.open(self.prefix + row["image_path"])).to(self.device)
        text = self.recipes[index].to(self.device)

        return image, text
    
    def __len__(self):
        return len(self.data)