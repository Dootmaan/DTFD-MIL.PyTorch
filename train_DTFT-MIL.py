from model.network import Classifier_1fc, DimReduction
from model.Attention import Attention_Gated as Attention
from model.Attention import Attention_with_Classifier
import argparse
import torch
from dataset.EmbededFeatsDataset import EmbededFeatsDataset
# torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import roc_auc_score
import numpy as np
parser = argparse.ArgumentParser(description='abc')

parser.add_argument('--name', default='abc', type=str)
parser.add_argument('--EPOCH', default=200, type=int)
parser.add_argument('--epoch_step', default='[100]', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--isPar', default=False, type=bool)
parser.add_argument('--log_dir', default='./debug_log', type=str)   ## log file path
parser.add_argument('--train_show_freq', default=40, type=int)
parser.add_argument('--droprate', default='0', type=float)
parser.add_argument('--droprate_2', default='0', type=float)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--batch_size_v', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_cls', default=2, type=int)
parser.add_argument('--mDATA0_dir_train0', default='', type=str)  ## Train Set
parser.add_argument('--mDATA0_dir_val0', default='', type=str)      ## Validation Set
parser.add_argument('--mDATA_dir_test0', default='', type=str)         ## Test Set
parser.add_argument('--numGroup', default=4, type=int)
parser.add_argument('--total_instance', default=4, type=int)
parser.add_argument('--numGroup_test', default=4, type=int)
parser.add_argument('--total_instance_test', default=4, type=int)
parser.add_argument('--mDim', default=512, type=int)
parser.add_argument('--grad_clipping', default=5, type=float)
parser.add_argument('--isSaveModel', action='store_false')
parser.add_argument('--debug_DATA_dir', default='', type=str)
parser.add_argument('--numLayer_Res', default=0, type=int)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--num_MeanInference', default=1, type=int)
parser.add_argument('--distill_type', default='AFS', type=str)   ## MaxMinS, MaxS, AFS
params = parser.parse_args()

classifier = Classifier_1fc(params.mDim, params.num_cls, params.droprate).to(params.device)
attention = Attention(params.mDim).to(params.device)
dimReduction = DimReduction(1024, params.mDim, numLayer_Res=params.numLayer_Res).to(params.device)
attCls = Attention_with_Classifier(L=params.mDim, num_cls=params.num_cls, droprate=params.droprate_2).to(params.device)

pretrained_weights=torch.load('model_best.pth')
classifier.load_state_dict(pretrained_weights['classifier'])
dimReduction.load_state_dict(pretrained_weights['dim_reduction'])
attention.load_state_dict(pretrained_weights['attention'])
attCls.load_state_dict(pretrained_weights['att_classifier'])

trainset=EmbededFeatsDataset('/newdata/why/CAMELYON16/',mode='train')
valset=EmbededFeatsDataset('/newdata/why/CAMELYON16/',mode='val')
testset=EmbededFeatsDataset('/newdata/why/CAMELYON16/',mode='test')

trainloader=torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, drop_last=False)
valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, drop_last=False)
testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, drop_last=False)

classifier.train()
dimReduction.train()
attention.train()
attCls.train()

trainable_parameters = []
trainable_parameters += list(classifier.parameters())
trainable_parameters += list(attention.parameters())
trainable_parameters += list(dimReduction.parameters())

optimizer0 = torch.optim.Adam(trainable_parameters, lr=params.lr,  weight_decay=params.weight_decay)
optimizer1 = torch.optim.Adam(attCls.parameters(), lr=params.lr,  weight_decay=params.weight_decay)

scheduler0 = torch.optim.lr_scheduler.MultiStepLR(optimizer0, [100], gamma=params.lr_decay_ratio)
scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer1, [100], gamma=params.lr_decay_ratio)

best_auc = 0
best_epoch = -1
test_auc = 0

ce_cri = torch.nn.CrossEntropyLoss(reduction='none').to(params.device)

def TestModel(test_loader):
    classifier.eval()
    dimReduction.eval()
    attention.eval()
    attCls.eval()

    all_pred=[]
    all_gt=[]
    for i, data in enumerate(test_loader):
        inputs, labels=data

        all_gt.append(labels.item())
        # labels=labels.to(params.device)

        # slide_sub_preds=[]
        # slide_sub_labels=[]
        slide_pseudo_feat=[]
        inputs_pseudo_bags=torch.chunk(inputs.squeeze(0), params.numGroup,dim=0)
        
        for subFeat_tensor in inputs_pseudo_bags:
            # slide_sub_labels.append(labels)
            subFeat_tensor=subFeat_tensor.to(params.device)
            # subFeat_tensor = torch.index_select(inputs_pseudo_bags, dim=0, index=torch.LongTensor(tindex).to(params.device))
            with torch.no_grad():
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            # with torch.no_grad():
            #     tPredict = classifier(tattFeat_tensor)  ### 1 x 2
            # slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        # slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        # slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
        # loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
        # optimizer0.zero_grad()
        # loss0.backward(retain_graph=True)
        # torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
        # torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
        # torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
        # optimizer0.step()

        ## optimization for the second tier
        gSlidePred = torch.softmax(attCls(slide_pseudo_feat), dim=1)
        all_pred.extend(gSlidePred.cpu().data.numpy().tolist())
        # loss1 = ce_cri(gSlidePred, labels)
        # optimizer1.zero_grad()
        # loss1.backward()
        # torch.nn.utils.clip_grad_norm_(attCls.parameters(), params.grad_clipping)
        # optimizer1.step()
    all_pred=np.array(all_pred)
    auc=roc_auc_score(all_gt, all_pred[:,1])
    acc=np.sum(all_gt==np.argmax(all_pred,1))/len(all_gt)
    print('auc{},acc:{}'.format(auc,acc))
    return auc,acc

best_auc=0.7
TestModel(testloader)
for ii in range(params.EPOCH):

    for param_group in optimizer1.param_groups:
        curLR = param_group['lr']
        print('current learning rate {}'.format(curLR))

    classifier.train()
    dimReduction.train()
    attention.train()
    attCls.train()

    # instance_per_group = total_instance // numGroup

    # numSlides = len(SlideNames_list)
    # numIter = numSlides // params.batch_size

    # tIDX = list(range(numSlides))
    # random.shuffle(tIDX)

    for i, data in enumerate(trainloader):
        inputs, labels=data
        labels=labels.to(params.device)

        slide_sub_preds=[]
        slide_sub_labels=[]
        slide_pseudo_feat=[]
        inputs_pseudo_bags=torch.chunk(inputs.squeeze(0), params.numGroup,dim=0)

        for subFeat_tensor in inputs_pseudo_bags:

            slide_sub_labels.append(labels)
            subFeat_tensor=subFeat_tensor.to(params.device)
            # subFeat_tensor = torch.index_select(inputs_pseudo_bags, dim=0, index=torch.LongTensor(tindex).to(params.device))
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)
            slide_pseudo_feat.append(tattFeat_tensor)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0) ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0) ### numGroup
        loss0 = ce_cri(slide_sub_preds, slide_sub_labels).mean()
        optimizer0.zero_grad()
        loss0.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), params.grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), params.grad_clipping)
        optimizer0.step()

        ## optimization for the second tier
        gSlidePred = attCls(slide_pseudo_feat)
        loss1 = ce_cri(gSlidePred, labels)
        optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(attCls.parameters(), params.grad_clipping)
        optimizer1.step()

        if i%10==0:
            print('[EPOCH{}:ITER{}] loss0:{}; loss1:{}'.format(ii,i,loss0.item(),loss1.item()))
    
    scheduler0.step()
    scheduler1.step()
    
    auc,acc=TestModel(valloader)
    if auc>best_auc:
        best_auc=auc
        print('new best auc. Testing...')
        TestModel(testloader)
        tsave_dict = {
            'classifier': classifier.state_dict(),
            'dim_reduction': dimReduction.state_dict(),
            'attention': attention.state_dict(),
            'att_classifier': attCls.state_dict()
        }
        torch.save(tsave_dict, 'model_best.pth')
