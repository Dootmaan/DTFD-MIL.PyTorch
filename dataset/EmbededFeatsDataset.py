import torch
import glob
import random
import numpy as np
import pandas

class EmbededFeatsDataset(torch.utils.data.Dataset):
    def __init__(self,path,mode='train'):
        super().__init__()
        self.mode=mode
        self.data=[]
        self.label=[]
        if mode=='train' or self.mode=='val':
            filenames=sorted(glob.glob(path+'/extracted_patches/training/*/256.0/*/feats1024.npy'))
            random.seed(552) # make sure each time we have the same filenames order.
            random.shuffle(filenames)
            random.seed()
            train_frac, val_frac = 0.8, 0.2
            n_train=int(train_frac*len(filenames)+1)
            n_val=int(len(filenames)-n_train)
            
            if mode=='train':
                for i in range(n_train):
                    fname=filenames[i]
                    print('processing:',fname)
                    npy=np.load(fname)
                    self.data.append(npy)

                    label=fname.split(r'/')[-4]
                    if label=='tumor':
                        self.label.append(1)
                    if label=='normal':
                        self.label.append(0)

            if mode=='val':
                for i in range(n_train,n_train+n_val):
                    fname=filenames[i]
                    print('processing:',fname)
                    npy=np.load(fname)
                    self.data.append(npy)

                    label=fname.split(r'/')[-4]
                    if label=='tumor':
                        self.label.append(1)
                    if label=='normal':
                        self.label.append(0)

        if mode=='test':
            filenames=sorted(glob.glob(path+'/extracted_patches/testing/*/256.0/*/feats1024.npy'))
            for fname in filenames:
                print('processing:',fname)
                npy=np.load(fname)
                self.data.append(npy)

                case=fname.split(r'/')[-2]
                labels=pandas.read_csv(path+'/testing/reference.csv',index_col=0,header=None)
                label=labels.loc[case,1]

                if label=='Tumor':
                    self.label.append(1)
                if label=='Normal':
                    self.label.append(0)

    def __len__(self):
        return len(self.label)

    def augment(self,feats, base=0.95):
        np.random.shuffle(feats)
        length=len(feats)
        rand_len=random.randint(int(base*length),length)
        feats=feats[:rand_len,:]
        return feats

    def __getitem__(self,index):
        # npy=np.load(self.data[index])
        return self.data[index],self.label[index]

if __name__=="__main__":
    dataset=EmbededFeatsDataset('/newdata/why/CAMELYON16/',mode='val')
    dataset.__getitem__(40)
    dataset=EmbededFeatsDataset('/newdata/why/CAMELYON16/',mode='test')