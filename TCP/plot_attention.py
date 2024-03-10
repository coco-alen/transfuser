import os
import pickle

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from config import GlobalConfig
from data_new import CARLA_Data
from model_transformer import VitFuser, SelfAttention
from utils import load_weight

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = VitFuser(config, "cpu")
    
    def forward(self, data):
        input = ([data["front_img"].unsqueeze(0)],
                [data["front_img"].unsqueeze(0)],
                torch.tensor(data["target_point"],dtype=torch.float).unsqueeze(0),
                torch.tensor(data["speed"],dtype=torch.float).view(1,-1),
                torch.tensor(data["target_command"],dtype=torch.int).unsqueeze(0))
        return self.model(*input)


def get_attention_hook(module, input, output):
    # 假设attention权重是输出的第二个元素
    attention_weights = output[1].detach().cpu().numpy()
    attention.append(attention_weights)

def find_and_register_hook(module, hook_function):
    if isinstance(module, SelfAttention):
        module.register_forward_hook(hook_function)
    else:
        for child in module.children():
            find_and_register_hook(child, hook_function)

attention = []
config = GlobalConfig()
val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data)
model = Model()
weights = torch.load("/home/gyp/program/my_transfuser/transfuser/TCP/log/vitfuser/best_epoch=52-val_loss=0.784.ckpt")
load_weight(model, weights["state_dict"], strict=True)
find_and_register_hook(model, hook_function=get_attention_hook)
data = val_set[653]
output = model(data)
with open("attention.pickle", "wb") as f:
    pickle.dump(attention, f)

def visualize_attention(attention_weights, path):
    # 将attention权重转换为numpy数组
    # 使用seaborn库创建热图
    plt.figure(figsize=(12,10))
    ax = sns.heatmap(attention_weights, cmap='viridis')
    ax.axvline(15, color='red')  # 在第15列画上红线
    ax.axvline(31, color='red')
    ax.axvline(34, color='red')

    ax.axhline(15, color='red')  # 在第15列画上红线
    ax.axhline(31, color='red')
    ax.axhline(34, color='red')
    ax.axis('off')

    ax.invert_xaxis()

    ax.text(12, 42, 'main image', fontsize=18)
    ax.text(27, 42, 'side image', fontsize=18)
    ax.text(35, 42, 'measure', fontsize=18)
    ax.text(40, 42, 'predict', fontsize=18)

    ax.text(46, 9, 'main\nimage', fontsize=18)
    ax.text(46, 25, 'side\nimage', fontsize=18)
    ax.text(46, 33, 'measure', fontsize=18)
    ax.text(46, 38, 'predict', fontsize=18)

    plt.savefig(path)
    plt.close()

with open("attention.pickle", "rb") as f:
    attention = pickle.load(f)

os.makedirs("attention_map", exist_ok=True)

for i in tqdm(range(len(attention))):
    attention_value = attention[i][0]
    for j in range(attention_value.shape[0]):
        visualize_attention(attention_value[j], f"attention_map/{i}_{j}.png")
    visualize_attention(np.sum(attention_value,axis=0), f"attention_map/{i}.png")

attention_value = np.sum(sum(attention)[0],axis=0)
visualize_attention(attention_value, f"attention_map/sum.png")