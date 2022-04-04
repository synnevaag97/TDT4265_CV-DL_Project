from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def get_config(config_path):
    cfg = LazyConfig.load(config_path)
    cfg.train.batch_size = 1
    return cfg


def get_dataloader(cfg, dataset_to_visualize):
    if dataset_to_visualize == "train":
        # Remove GroundTruthBoxesToAnchors transform
        cfg.data_train.dataset.transform.transforms = cfg.data_train.dataset.transform.transforms[:-1]
        data_loader = instantiate(cfg.data_train.dataloader)
    else:
        cfg.data_val.dataloader.collate_fn = utils.batch_collate
        data_loader = instantiate(cfg.data_val.dataloader)

    return data_loader


def analyze_something(dataloader, cfg):
    box_ratio = np.empty(0)
    min_size = np.empty(0)
    wid = np.empty(0)
    hei = np.empty(0)
    x = np.empty(0)
    y = np.empty(0)
    
    number = 0
    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        #print("The boxes are :", batch['boxes'][0], "The id is: ", batch['image_id']) #['image', 'boxes', 'labels', 'width', 'height', 'image_id']
        
        #Histograms of the ancor boxes' ratios and min_size
        # boxes type [x,y, width, height] 
        # x, y: the upper-left coordinates of the bounding box width, height: the dimensions of your bounding box
        
        
        widths = (batch['boxes'][0][:,2].numpy()-batch['boxes'][0][:,0].numpy())*batch['width'].numpy()
        heights = (batch['boxes'][0][:,3].numpy()-batch['boxes'][0][:,1].numpy())*batch['height'].numpy()
        x = np.append(x,batch['boxes'][0][:,0].numpy()*batch['width'].numpy())
        y = np.append(y,batch['boxes'][0][:,1].numpy()*batch['height'].numpy())
        #widths = batch['boxes'][0][:,2].numpy()*batch['width'].numpy()
        #heights = batch['boxes'][0][:,3].numpy()*batch['height'].numpy()
              
        #print("the width is: ",widths)
        
        ratio = widths/heights
        
        indices1 = [i for i, x in enumerate(ratio) if x > 1] #indices of the ratios that are bigger than 1
        indices2 = [i for i, x in enumerate(ratio) if x < 1] #indices of the ratios that are smaller than 1
        
        
        sizes = widths[indices1]/np.sqrt(ratio[indices1])
        sizes = np.append(sizes, heights[indices2]/np.sqrt(ratio[indices2]))
        
        box_ratio = np.append(box_ratio,ratio)
        min_size = np.append(min_size,sizes)
        
        wid = np.append(wid,widths)
        hei = np.append(hei,heights)
        
        #number += 1
        if number == 5:
            break    # break here
        
    box_ratio[box_ratio < 1] = 1/box_ratio[box_ratio < 1]
    
    fig = plt.figure() 
    plt.hist(box_ratio, bins=50)
    fig.savefig('dataset_exploration/histo_ratios.png')
    
    fig = plt.figure() 
    plt.hist(min_size, bins=250)
    #plt.xlim([0, 250])
    fig.savefig('dataset_exploration/histo_min_size.png')
    #plt.show()
    
    fig = plt.figure() 
    plt.hist(wid, bins=300)
    plt.xlim([0, 40])
    fig.savefig('dataset_exploration/histo_width.png')
    
    fig = plt.figure() 
    plt.hist(hei, bins=100)
    plt.xlim([0, 70])
    fig.savefig('dataset_exploration/histo_heights.png')
    
    fig = plt.figure() 
    plt.hist(x, bins=100)
    fig.savefig('dataset_exploration/histo_x.png')
    
    fig = plt.figure() 
    plt.hist(y, bins=50)
    fig.savefig('dataset_exploration/histo_y.png')
    
    
        

def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)
    
    


if __name__ == '__main__':
    main()
