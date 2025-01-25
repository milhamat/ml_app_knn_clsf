import os
from src.models.train import TrainModel
from src.gui.ui import Interface



file_path = os.path.join("artifacts", "models", "model.pkl")

if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
    ui = Interface()
    ui.lunch()
else:
    print(f"The file '{file_path}' does not exist.")
    train = TrainModel()
    train.train()