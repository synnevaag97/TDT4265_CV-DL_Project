from tops.config import instantiate, LazyConfig
from ssd import utils
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import ListedColormap

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
    # Init
    box_ratio = np.empty(0)
    min_size = np.empty(0)
    wid = np.empty(0)
    hei = np.empty(0)
    x = np.empty(0)
    y = np.empty(0)
    number = 0
    
    count = torch.zeros(9)
    box_ratioc = [[-5]] * 9
    widc = [[-5]] * 9
    heic = [[-5]] * 9
    min_sizec = [[-5]] * 9
    
    list_val = np.array([[-5,-5,-5,-5,-5]])

    for batch in tqdm(dataloader):
        # Remove the two lines below and start analyzing :D
        #print("The boxes are :", batch['boxes'][0], "The id is: ", batch['image_id']) #['image', 'boxes', 'labels', 'width', 'height', 'image_id']
        
        #Histograms of the ancor boxes' ratios and min_size
        # boxes type [x,y, width, height] 
        # x, y: the upper-left coordinates of the bounding box width, height: the dimensions of your bounding box
         
        
        lab = batch['labels'][0]
        count += torch.bincount(lab, minlength=9)
        for i,box in enumerate(batch['boxes'][0]):
            #print(i, "boxes",box)
            
            width = (box[2].numpy() - box[0].numpy())*batch['width'].numpy()[0]
            height = (box[3].numpy() - box[1].numpy())*batch['height'].numpy()[0]
            
            #print("the width is: ",widths)

            ratio = width/height
            
            if ratio > 1:
                size = width/np.sqrt(ratio)
            else:
                size = height*np.sqrt(ratio)

            min_sizec[lab[i].item()] = np.append(min_sizec[lab[i].item()],size)
            
            if ratio < 1:
                ratio = 1/ratio

            box_ratioc[lab[i].item()] = np.append(box_ratioc[lab[i].item()],ratio)
            widc[lab[i].item()] = np.append(widc[lab[i].item()],width)
            heic[lab[i].item()]= np.append(heic[lab[i].item()],height)
        
            #print("size:  ",size)
            ls = np.array([[size, ratio, width, height, lab[i].item()]])
            list_val = np.append(list_val,ls,axis=0)
        """
        number += 1
        if number == 5:
            break    # break here
        
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
        #if number == 5:
            #break    # break here
        
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
    plt.hist(min_size, bins=250)
    plt.xlim([0, 35])
    fig.savefig('dataset_exploration/histo_min_size_small.png')
    #plt.show()
    
    fig = plt.figure() 
    plt.hist(wid, bins=300)
    plt.xlim([0, 40])
    fig.savefig('dataset_exploration/histo_width.png')
    
    fig = plt.figure() 
    plt.hist(hei, bins=100)
    plt.xlim([0, 70])
    fig.savefig('dataset_exploration/histo_heights.png')
"""

        
    list_val = np.delete(list_val, 0, axis = 0)
    list_val = list_val[list_val[:,0].argsort()]
    for i in range(1,9):
        fig = plt.figure() 
        plt.hist(box_ratioc[i], bins=20)
        plt.xlim([0, 10])
        fig.savefig('dataset_exploration/Histograms/histo_ratios_'+str(i) +'.png')

        fig = plt.figure() 
        plt.hist(min_sizec[i], bins=20)
        plt.xlim([0, 400])
        fig.savefig('dataset_exploration/Histograms/histo_min_size_'+str(i) +'.png')

        fig = plt.figure() 
        plt.hist(widc[i], bins=50)
        plt.xlim([0, 1024])
        fig.savefig('dataset_exploration/Histograms/histo_width_'+str(i) +'.png')

        fig = plt.figure() 
        plt.hist(heic[i], bins=50)
        plt.xlim([0, 128])
        fig.savefig('dataset_exploration/Histograms/histo_heights_'+str(i) +'.png')
        plt.close('all')
    
    fig = plt.figure() 
    plt.hist(list_val[:,0], bins=400)
    plt.xlim([0, 60])
    fig.savefig('dataset_exploration/Histograms/histo_small.png')
    plt.close('all')

    fig = plt.figure() 
    plt.hist(list_val[:,0], bins=400)
    plt.xlim([0, 130])
    fig.savefig('dataset_exploration/Histograms/histo_total.png')
    plt.close('all')
        
    print("The number of occurences for every class is: ",count)
    fig = plt.figure() 
    classes = ['car','truck','bus','bicycle','scooter','person','rider']
    s = [5 for n in range(len(list_val[:,0]))]
    scatter = plt.scatter(list_val[:,0],list_val[:,1],s = s, cmap = 'Set1', c=list_val[:,4])
    plt.ylim([0, 7])
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc='best')
    fig.savefig('dataset_exploration/Histograms/plot.png', dpi = 150)
    
    plt.close('all')
    
    fig = plt.figure() 
    classes = ['car','truck','bus','bicycle','scooter','person','rider']
    s = [5 for n in range(len(list_val[:,0]))]
    scatter = plt.scatter(list_val[:,2],list_val[:,3],s = s, cmap = 'Set1', c=list_val[:,4])
    #plt.ylim([0, 7])
    plt.legend(handles=scatter.legend_elements()[0], labels=classes, loc='best')
    fig.savefig('dataset_exploration/Histograms/plot_height_width.png', dpi = 150)
    
    plt.close('all')
    
    print((list_val[:,1]<2))
    print(np.bitwise_and(list_val[:,1]<2, list_val[:,0]<10))
    print("The number of object with ratio smaller than 2 is: ", np.count_nonzero(np.bitwise_and(list_val[:,1]<2, list_val[:,0]<20, list_val[:,0]>10)))
    print("The number of object with ratio bigger than 2 is: ", np.count_nonzero(np.bitwise_and(list_val[:,1]>=2, list_val[:,0]<20, list_val[:,0]>10)))
    print("The number of object with ratio smaller than 3 is: ", np.count_nonzero(np.bitwise_and(list_val[:,1]<3, list_val[:,0]<20, list_val[:,0]>10)))
    print("The number of object with ratio bigger than 3 is: ", np.count_nonzero(np.bitwise_and(list_val[:,1]>=3, list_val[:,0]<20, list_val[:,0]>10)))
    print("The number of object with ratio smaller than 4 is: ", np.count_nonzero(np.bitwise_and(list_val[:,1]<4, list_val[:,0]<20, list_val[:,0]>10)))
    print("The number of object with ratio bigger than 4 is: ", np.count_nonzero(np.bitwise_and(list_val[:,1]>=4, list_val[:,0]<20, list_val[:,0]>10)))
    print("The mean ratio is: ", np.mean(list_val[list_val[:,0]<10,1]))
    print("The smallest object has size: ", list_val[1:50,0])
    print("The biggest objects has size: ", list_val[len(list_val)-50:len(list_val),0])
def main():
    config_path = "configs/tdt4265.py"
    cfg = get_config(config_path)
    dataset_to_analyze = "train"  # or "val"

    print("Label map is:", cfg.label_map)

    dataloader = get_dataloader(cfg, dataset_to_analyze)
    analyze_something(dataloader, cfg)
    
if __name__ == '__main__':
    main()

"""
The number of occurences for every class is:  tensor([   0., 6922.,   84.,  251.,    0.,  805.,  465., 3892., 1209.])
The number of object with ratio smaller than 4 is:  11564
The number of object with ratio bigger than 4 is:  2064
"""