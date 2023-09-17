import torch
import os
import random 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pickle
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


from network import Encoder, Classifier, Discriminator
from ImageLoader import Dataloder_img



def shuffler(h1, h2):

    sh = torch.randint(0, 2, (h1.size(0),))  # 1 means keeping h1 and 0 means keeping h2
  
    h1new = torch.matmul(torch.diag(1-sh).float(), h2.cpu())+torch.matmul(torch.diag(sh).float(),h1.cpu())
    h2new = torch.matmul(torch.diag(1-sh).float(), h1.cpu())+torch.matmul(torch.diag(sh).float(),h2.cpu())

    return h1new.cuda(), h2new.cuda(), sh


# Hyperparameter Details
epochs = 10
hash_code = 48
batch_size = 32
learningRate = 0.0001
gamma = 1 #cauchy probablity scale
alpha1 = 0.5 # hyperparameter cauchy loss 1
alpha2 = 0.6 # hyperparameter cauchy loss 2
beta = 0.5 ## hyperparameter discriminator loss


trainingDataPath = "./data/train"
numClasses = len(os.listdir(trainingDataPath))
print('Num. of classes:', numClasses)

# Model Intilization
encoder = Encoder(hash_code)
classifier = Classifier(numClasses)
discriminator = Discriminator(hash_code)
if torch.cuda.is_available():
    encoder.cuda()
    classifier.cuda()
    discriminator.cuda()

#model_path = './models/OPHash_48.pkl'
#encoder.load_state_dict(torch.load(model_path), strict=False)

transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=2),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

trainset = Dataloder_img(trainingDataPath, transform=transform, target_transform=None)
print(len(trainset))
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True,  batch_size=batch_size, num_workers=4)

print("\nDataset generated. \n\n")

#loss metrics
criterion = nn.BCELoss()
ac1_loss_criterion = nn.NLLLoss()
discriminator_criterion = nn.BCEWithLogitsLoss()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learningRate, eps=0.0001, amsgrad=True)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=learningRate, eps=0.0001, amsgrad=True)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr = learningRate/10,eps = 0.0001,amsgrad = True)


hd_item = {}

for epoch in tqdm(range(epochs)):
    print ("Epoch:%d/%s."%(epoch+1,epochs))


    hd_t0 = 0
    hd_t1 = 0
    hd_t2 = 0

    for i, data in enumerate(trainloader, 0):
        #print(i)
        # if i > 4 :
            # break
        input1, input2, labels, groundtruths1, groundtruths2 = data #image and label calling

        indexes0 = np.where(labels.numpy() == 0)[0].tolist() #type 1
        indexes1 = np.where(labels.numpy() == 1)[0].tolist() #type 2
        indexes2 = np.where(labels.numpy() == 2)[0].tolist() #type 3

        input1, input2, labels = Variable(input1).cuda(), Variable(input2).cuda(), Variable(labels).cuda()
        groundtruths1, groundtruths2 = Variable(groundtruths1).cuda(), Variable(groundtruths2).cuda()
        #print(input1.shape)
        h1, x1 = encoder(input1)
        h2, x2 = encoder(input2)
        #print(h1.shape)
        #print(x2.shape)
        
        
        #################################
        ##### AUXILARY CLASSIFIER1  #####
        #################################
        pred = classifier(x1)
        ac1_loss =  ac1_loss_criterion(pred, groundtruths1)

        classifier_optimizer.zero_grad()
        encoder_optimizer.zero_grad()

        ac1_loss.backward()

        classifier_optimizer.step()
        encoder_optimizer.step()

        
        ###########################################
        ##### CAUCHY LOSS 1 [t =2 vs t=1,t=0] #####
        ###########################################
        h1, x1 = encoder(input1)
        h2, x2 = encoder(input2)
 
            
        torch.autograd.set_detect_anomaly(True)
        s = (labels != 2) #similarity indexes of (s_{i,j})
        cos = F.cosine_similarity(h1, h2, dim=1, eps=1e-6)
        dist = F.relu((1-cos)*hash_code/2)


        
        #(dist)
        hd_t0 += torch.sum(dist[indexes0]).item()/(dist[indexes0].size(0) + 0.0000001)
        hd_t1 += torch.sum(dist[indexes1]).item()/(dist[indexes1].size(0) + 0.0000001)
        hd_t2 += torch.sum(dist[indexes2]).item()/(dist[indexes2].size(0) + 0.0000001)
        #print(hd_t2)
        cauchy_output = torch.reciprocal(dist+gamma)*gamma
        try:
            loss1 = alpha1*criterion(torch.squeeze(cauchy_output), s.float())
        except RuntimeError:
            print(torch.squeeze(cauchy_output))
            print(s)
            print("s", torch.max(s.float()).item(),torch.min(s.float()).item())
            print("\nCO ", torch.max(torch.squeeze(cauchy_output)).item(), torch.min(torch.squeeze(cauchy_output)).item())


        #######################################
        #####  CAUCHY LOSS 2 [t=1 vs t=0] #####
        #######################################

        if not len(indexes2) == batch_size:
            input1_new = torch.from_numpy(np.delete(input1.cpu().data.numpy(), indexes2, 0))
            input2_new = torch.from_numpy(np.delete(input2.cpu().data.numpy(), indexes2, 0))
            labels_2 = 1-labels[labels != 2] #relational indexes of (r_{i,j})
            input1_new, input2_new, labels_2 = Variable(input1_new).cuda(), Variable(input2_new).cuda(), Variable(labels_2).cuda()
            h1_new, _ = encoder(input1_new)
            #print(input2_new.shape)
            h2_new, _ = encoder(input2_new)
            cos2 = F.cosine_similarity(h1_new, h2_new, dim=1, eps=1e-6)
            dist2 = F.relu((1-cos2)*hash_code/2)
            cauchy_output2 = torch.reciprocal(dist2+gamma)*gamma
            try:
                loss2 = beta*criterion((cauchy_output2), labels_2.float())
            except RuntimeError:
                print(torch.squeeze(cauchy_output2))
                print(labels_2)
                print("s", torch.max(labels_2.float()).item(), torch.min(labels_2.float()).item())
                print("\nCO ", torch.max(torch.squeeze(cauchy_output2)).item(), torch.min(torch.squeeze(cauchy_output2)).item())

        else:
            print('-------------------No images for type 0 and type 1--------------')

 
        loss=  loss1 + loss2
        loss.backward(retain_graph = True)
        encoder_optimizer.step()
  
        #########################
        ##### Discriminator #####
        #########################
        h1, x1 = encoder(input1)
        h2, x2 = encoder(input2)
        #print('2')
        if len(indexes0) > 0: #applied on type 1
            d_h1 = h1[indexes0]
            d_h2 = h2[indexes0]
            d_h1, d_h2, dlabels = shuffler(d_h1, d_h2)
            dlabels = Variable(dlabels).cuda()
            d_input = torch.stack((d_h1, d_h2), 1)
            d_output = discriminator(d_input).view(-1)
            #print(d_output.shape)
            d_loss = beta*discriminator_criterion(d_output, dlabels.float())


            discriminator_optimizer.zero_grad()
            d_loss.backward()
            discriminator_optimizer.step()
            encoder_optimizer.step()


        del(input1)
        del(input2)
        

    dataStorePath =  './models/'

    hd_item[epoch] = (hd_t0/i, hd_t1/i, hd_t2/i)
    print('hamming distance:', hd_item[epoch])
    hd_path = os.path.join(dataStorePath, 'OPHash_hd_log.pkl')
    '''with open(hd_path, 'wb') as handle:
        pickle.dump(hd_item, handle)
        print("Saving hamming distance log to ", hd_path)'''

    encoder_path = os.path.join(dataStorePath, f'OPHash_{hash_code}.pkl')
    #torch.save(encoder.state_dict(), encoder_path)
    print("Saving model to ", encoder_path)
    print('------------------model saved-----------------------')
    print('------------------------------------------------------------------------------------------------')
