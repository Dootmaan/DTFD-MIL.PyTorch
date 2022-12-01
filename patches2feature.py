from model.feature_extraction import resnet50_baseline
import os
import glob
import torch
import numpy as np
import cv2

BATCH_SIZE=448
bag_nums=5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=resnet50_baseline(pretrained=True).to(device)
model=torch.nn.DataParallel(model)
model.eval()
filenames=sorted(glob.glob('/newdata/why/CAMELYON16/extracted_patches/*/*/256.1/*'))
for cases in filenames:
    print('processing:',cases)
    feats=[]
    imgs=sorted(glob.glob(cases+'/*.jpg'))
    for i in range(0,len(imgs),BATCH_SIZE):
        imgnames=imgs[i:i+BATCH_SIZE]
        input_tensor=[]
        for imgname in imgnames:
            img=cv2.imread(imgname).transpose(2,0,1)/255.0
            input_tensor.append(img)
        input_tensor=np.array(input_tensor)
        input_tensor=torch.autograd.Variable(torch.from_numpy(input_tensor)).type(torch.FloatTensor).to(device)
        with torch.no_grad():
            feat=model(input_tensor)
        feats.extend(feat.cpu().data.numpy())
    feats=np.array(feats)
    np.save(cases+'/feats1024.npy',feats)

