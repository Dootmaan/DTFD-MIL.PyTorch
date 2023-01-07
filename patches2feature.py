from model.feature_extraction import resnet50_baseline
# import os
import glob
import torch
import numpy as np
# import cv2
import torchvision.transforms as T
import PIL

BATCH_SIZE=448
# bag_nums=5
transform=T.Compose([
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=resnet50_baseline(pretrained=True).to(device)
# model=torch.nn.DataParallel(model)
model.eval()
filenames=sorted(glob.glob('/newdata/why/CAMELYON16/extracted_patches_0.8/*/*/256.1/*'))
for cases in filenames:
    print('processing:',cases)
    feats=[]
    imgs=sorted(glob.glob(cases+'/*.png'))
    for i in range(0,len(imgs),BATCH_SIZE):
        imgnames=imgs[i:i+BATCH_SIZE]
        input_tensor=torch.FloatTensor().to(device)
        for imgname in imgnames:
            img=PIL.Image.open(imgname)
            input_tensor=torch.cat([input_tensor,transform(img).to(device).unsqueeze(0)],dim=0)
        # input_tensor=torch.autograd.Variable(torch.from_numpy(input_tensor)).type(torch.FloatTensor).to(device)
        with torch.no_grad():
            feat=model(input_tensor)
        feats.extend(feat.cpu().data.numpy())
    feats=np.array(feats)
    np.save(cases+'/norm_feats1024.npy',feats)
