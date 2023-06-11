import torch
import os
import numpy as np
import random
import operator
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(0)
use_cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', use_cuda)
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

from torchvision import transforms
from PIL import Image

from network import Encoder
from metrics import mAP, nDCG


random.seed(3407)
np.random.seed(3407)
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.cuda.manual_seed_all(3407)


def relevenceClasses(sorted_pool,q_name):
    value = []
    for i in range(len(sorted_pool)):
        if ((q_name.split("_")[0]+'_'+q_name.split("_")[1]) ==(sorted_pool[i][0].split("_")[0]+'_'+sorted_pool[i][0].split("_")[1]) or (q_name.split("_")[2] == sorted_pool[i][0].split("_")[2])):
        
            value.append(1)
        elif (q_name.split("_")[1:3])== (sorted_pool[i][0].split("_")[1:3]):
            value.append(2)
        else:
            value.append(0)
    value2 = sorted(value, reverse=True)
    return value, value2

def hammingDistance(h1, h2):
    hash_code = h1.shape[1]
    h1norm = torch.div(h1, torch.norm(h1, p=2))
    h2norm = torch.div(h2, torch.norm(h2, p=2))
    distH = torch.pow(torch.norm(h1norm - h2norm, p=2), 2) * hash_code / 4
    return distH



#### Hyperparemetr Details ######
hash_code = 48
#model load######################

model = Encoder(hash_code)

if torch.cuda.is_available():
    model.cuda()

model_name = f'OPHash_{hash_code}.pkl'
dataStorePath = './models/'
model_path = os.path.join(dataStorePath,model_name)
model.load_state_dict(torch.load(model_path))

print(model_path)
galleryfolderpath = "./data/gallery"
queryfolderpath = "./data/query"
gallery_files = os.listdir(galleryfolderpath)
gallery_files = random.sample(gallery_files, len(gallery_files))
query_files = os.listdir(queryfolderpath)
query_files = random.sample(query_files, len(query_files))
print(len(gallery_files))
querynumber = len((query_files))
print(querynumber)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

gallery = {}
print("\n\n Building Gallery .... \n")
with torch.no_grad():
    # Process each gallery image
    for img in gallery_files:
        image_path = os.path.join(galleryfolderpath, img)

        # Load and transform the image
        image = np.load(image_path)
        # transfer to one channel
        if len(image.shape)!= 2:
            image = np.mean(image,axis=-1)

        image = Image.fromarray(image)
        tensor_image = transform(image).unsqueeze(0).cuda()

        # Pass the tensor through the  model
        h, _ = model(tensor_image)

        # Store the result in the gallery dictionary
        gallery[img] = torch.sign(h)

        # Clean up
        del tensor_image
    print("\n Building Complete. \n")

    count = 0

    q_prec_10 = 0
    q_prec_100 = 0
    q_prec_1000 = 0
    
    nDCG_list_10 = []
    nDCG_list_100 = []
    nDCG_list_1000 = []

    #print(len(qNimage[0:100]))
    for q_name in query_files:
        if 'Chest_CT' not in q_name:
            #
            count = count+1
            query_image_path = os.path.join(queryfolderpath, q_name)
            # Load and transform the image
            query_image = np.load(query_image_path)
            # transfer to one channel
            if len(query_image.shape)!= 2:
                query_image = np.mean(query_image,axis=-1)
            query_image = Image.fromarray(query_image)
            query_tensor_image = transform(query_image).unsqueeze(0).cuda()

            # Pass the tensor through the model
            h_q, _ = model(query_tensor_image)
            h_q = torch.sign(h_q)
            dist = {}
            for key, h1 in gallery.items():
                dist[key] = hammingDistance(h1, h_q)

            print(count)   
            ### images with sorted distance 
            sorted_pool_10 = sorted(dist.items(), key=operator.itemgetter(1))[0:10]
            sorted_pool_100 = sorted(dist.items(), key=operator.itemgetter(1))[0:100]
            sorted_pool_1000 = sorted(dist.items(), key=operator.itemgetter(1))[0:1000]

            #### mean average precision
            q_prec_10 += mAP(q_name, sorted_pool_10)
            q_prec_100 += mAP(q_name, sorted_pool_100)
            q_prec_1000 += mAP(q_name, sorted_pool_1000)

            ### nDCG
            r_i_10, sorted_r_i_10 = relevenceClasses(sorted_pool_10, q_name)
            r_i_100, sorted_r_i_100 = relevenceClasses(sorted_pool_100, q_name)
            r_i_1000, sorted_r_i_1000 = relevenceClasses(sorted_pool_1000, q_name)
            #print(r_i, sorted_r_i)

            nDCG_value_10 = nDCG(r_i_10, sorted_r_i_10)
            nDCG_list_10.append(nDCG_value_10)

            nDCG_value_100 = nDCG(r_i_100, sorted_r_i_100)
            nDCG_list_100.append(nDCG_value_100)

            nDCG_value_1000 = nDCG(r_i_1000, sorted_r_i_1000)
            nDCG_list_1000.append(nDCG_value_1000)

            if count % 10 == 0:
                print("mAP@10 :", q_prec_10/count)
                print("mAP@100 :", q_prec_100/count)
                print("mAP@1000 :", q_prec_1000/count)
                print('-------------------------------')
                print('nDCG@10:', sum(nDCG_list_10)/len(nDCG_list_10))
                print('nDCG@100:', sum(nDCG_list_100)/len(nDCG_list_100))
                print('nDCG@1000:', sum(nDCG_list_1000)/len(nDCG_list_1000))


print('-----------------------------------------------')       
print("mAP@10 :", q_prec_10/count)
print("mAP@100 :", q_prec_100/count)
print("mAP@1000 :", q_prec_1000/count)
print('-------------------------------')
print('nDCG@10:', sum(nDCG_list_10)/len(nDCG_list_10))
print('nDCG@100:', sum(nDCG_list_100)/len(nDCG_list_100))
print('nDCG@1000:', sum(nDCG_list_1000)/len(nDCG_list_1000))
print(hash_code)
print(model_name)
