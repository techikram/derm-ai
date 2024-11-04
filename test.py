import os
import torch
import torch.nn as nn
import albumentations
import cv2
import sys
import geffnet
import pandas as pd
from datetime import datetime
import numpy as np
from torch.utils.data import Dataset
class SplitAttention(nn.Module):
    def __init__(self,channel,k=3):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) #bs,k,n,c
        a=torch.sum(torch.sum(x_all,1),1) #bs,c
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) #bs,kc
        hat_a=hat_a.reshape(b,self.k,c) #bs,k,c
        bar_a=self.softmax(hat_a) #bs,k,c
        attention=bar_a.unsqueeze(-2) # #bs,k,1,c
        out=attention*x_all # #bs,k,n,c
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out
def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x
def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x
class S2Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention(channels)

    def forward(self, x):
        b,c,w,h = x.size()
        x=x.permute(0,2,3,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all=torch.stack([x1,x2,x3],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,3,1,2)
        return x
class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output
class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output
class CBAMBlock(nn.Module):
    def __init__(self,channel,reduction=16,kernel_size=7):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual  
class S2MLP_CBAM(nn.Module):
    def __init__(self,  enet_type, out_dim, pretrained=False):
        super(S2MLP_CBAM, self).__init__()
        backbone_map = {
            'S2MLP_CBAM_B0': 'tf_efficientnet_b0',
            'S2MLP_CBAM_B1': 'tf_efficientnet_b1',
            'S2MLP_CBAM_B2': 'tf_efficientnet_b2',
            'S2MLP_CBAM_B3': 'tf_efficientnet_b5',
            'S2MLP_CBAM_B4': 'tf_efficientnet_b4',
            'S2MLP_CBAM_B5': 'tf_efficientnet_b5',
            'S2MLP_CBAM_B6': 'tf_efficientnet_b6',
            'S2MLP_CBAM_B7': 'tf_efficientnet_b7' }

        backbone = backbone_map.get(enet_type)
        #print(f'S2MLPCBAM model Using {backbone} backbone:')
        self.enet = geffnet.create_model(backbone, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features  
        self.conv_stem = self.enet.conv_stem 
        self.act1 = self.enet.act1
        self.bn1 = self.enet.bn1
        self.blocks = nn.ModuleList()
        self.blocks.append(self.enet.blocks[0])

        attn_ich = self.enet.blocks[0][0].conv_pw.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[1])
        attn_ich = self.enet.blocks[1][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[2])
        attn_ich = self.enet.blocks[2][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
 
        self.blocks.append(self.enet.blocks[3])
        attn_ich = self.enet.blocks[3][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[4])
        attn_ich = self.enet.blocks[4][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[5])

        attn_ich = self.enet.blocks[5][0].conv_pwl.out_channels
        self.blocks.append(S2Attention(channels=attn_ich))
        self.blocks.append(CBAMBlock(attn_ich))

        self.blocks.append(self.enet.blocks[6]) 
        attn_ich = self.enet.blocks[6][0].conv_pwl.out_channels   
        self.blocks.append(CBAMBlock(attn_ich))        
        self.conv_head =self.enet.conv_head 
        self.bn2 = self.enet.bn2
        self.act2 = self.enet.act2
        self.global_pool = self.enet.global_pool 
        attn_ich = self.conv_head.out_channels  
        self.classifier = nn.Linear(in_ch, out_dim)
        self.enet = nn.Identity()
   
    def extract(self, x):    
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for block in self.blocks:
            x = block(x) 
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)            
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x)
        x= x.squeeze(-1).squeeze(-1)  
        out1 = self.classifier(x)     
        return out1
class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]
     
    def get_labels(self):
        return torch.tensor(self.csv['target']).float()
    def __getitem__(self, index):

        #print("mod:",self.mode)
        row = self.csv.iloc[index]
        #print("path to image",row.filepath)
        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data          
        else:
            label = torch.tensor(self.csv.iloc[index].target).long()
            return data, label    
def get_transforms(image_size,get_test = False):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])
    
    transforms_test = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    
    transforms_val = transforms_train
    
    if get_test:
        return transforms_test

    return transforms_train, transforms_val
def get_trans(img, I):
    if I >= 4:
        img = img.transpose(2, 3)
    if I % 4 == 0:
        return img
    elif I % 4 == 1:
        return img.flip(2)
    elif I % 4 == 2:
        return img.flip(3)
    elif I % 4 == 3:
        return img.flip(2).flip(3)
    
def main(user_id):
    models = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, 'model')
    kernel = '12c_s2cbam_b4_512_512_35ep'
    image_size = 512
    data_path = os.path.join(script_dir, 'img', user_id)
    results_dir = os.path.join(script_dir, 'results', user_id)
    
    os.makedirs(results_dir, exist_ok=True)

    folds = [0,1,2,3,4]
    out_dim = 12
    n_test = 10
    data = []
    for fold in folds:
        model_file = os.path.join(model_dir, f'{kernel}_best_fold{fold}.pth')
        model = S2MLP_CBAM(
                enet_type= 'S2MLP_CBAM_B4',
                 out_dim= out_dim)     
        model = model.to(device)
        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file)
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)  
        model.eval()    
        models.append(model)  

    column_names = ['filename', 'filepath']
    df_test = pd.DataFrame(columns=column_names)
    for filename in os.listdir(data_path):
        if filename.endswith(('.png', '.jpg', '.jpeg','.JPG','PNG')):  
            img_path = os.path.join(data_path, filename)
            df_test.loc[len(df_test)] = [filename, img_path]

    transforms_test = get_transforms(image_size, get_test=True)
    dataset_test = MelanomaDataset(df_test, 'test', meta_features=None, transform=transforms_test) 
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=20, num_workers=4)

    PROBS = []
    with torch.no_grad():
        for (data) in test_loader:             
            data = data.to(device)
            probs = torch.zeros((data.shape[0], out_dim)).to(device)
            for model in models:
                for I in range(n_test):
                    l = model(get_trans(data, I))
                    probs += l.softmax(1)
            probs /= n_test
            probs /= len(models)
            PROBS.append(probs.detach().cpu())
    PROBS = torch.cat(PROBS).numpy()

    # Use the name of the first image for the CSV file
    first_image_filename = df_test['filename'].iloc[0]
    csv_name = os.path.splitext(first_image_filename)[0]

    # Create the results DataFrame for the current batch
    column_names = ['filename','Benign', 'Suspicious', 'Malignant', 'Melanoma']
    results = pd.DataFrame(columns=column_names)
    for i in range(len(PROBS)):
        ben = PROBS[i][0]+ PROBS[i][3]+ PROBS[i][4] + PROBS[i][5]+  PROBS[i][7]+ PROBS[i][8]+  PROBS[i][10] + PROBS[i][11]
        mal = PROBS[i][2]+ PROBS[i][6]+ PROBS[i][9]
        results.loc[len(results)] = [df_test['filename'][i], ben , PROBS[i][1] , mal, PROBS[i][9]]

    # Save the current results
    csv_path = os.path.join(results_dir, f"{csv_name}_results.csv")
    print(f"Guardando resultados en: {csv_path}")
    results.to_csv(csv_path, index=False)

    # Update the cumulative results CSV
    cumulative_csv_path = os.path.join(results_dir, "cumulative_results.csv")
    
    if os.path.exists(cumulative_csv_path):
        cumulative_results = pd.read_csv(cumulative_csv_path)
    else:
        cumulative_results = pd.DataFrame(columns=column_names)
    
    cumulative_results = pd.concat([cumulative_results, results], ignore_index=True)
    cumulative_results.to_csv(cumulative_csv_path, index=False)
    print(f"Resultados acumulativos guardados en: {cumulative_csv_path}")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = 'cuda-server-****'
    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1
    device = torch.device('cuda')
    user_id = sys.argv[1]
    main(user_id)