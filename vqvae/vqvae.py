import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchsummary import summary
import matplotlib.pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import numpy as np
import cv2
from tqdm.auto import tqdm
import os
import random

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    def __init__(self, backbone = 'resnet34'):
        super(Encoder, self).__init__()
        self.backbone = timm.create_model(backbone, pretrained = True)
        self.List = list(self.backbone.children())[:-4]
    def forward(self,X):
        for i,layer in enumerate(self.List):
            X = layer(X)
        return X                        
class discretize(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25):
        super(discretize, self).__init__()
        self.e_dim = e_dim
        self.n_e = n_e
        self.beta = beta
        self.code_book = nn.Embedding(n_e, e_dim)
        self.softmax = nn.Softmax(dim=1)
        self.code_book.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, enc):
        enc = enc.permute(0,2,3,1).contiguous()
        enc_flattened = enc.view(-1, self.e_dim)

        distances = (torch.sum(enc_flattened**2, dim=1, keepdim = True)+
                     torch.sum(self.code_book.weight**2, dim=1)
                        -2*torch.matmul(enc_flattened, self.code_book.weight.t()))
        min_encoding_ids = torch.argmin(distances, dim=1).unsqueeze(1)
        min_encodings_mask = torch.zeros(min_encoding_ids.shape[0], self.n_e, dtype = self.code_book.weight.dtype).to(enc.device)
        min_encodings_mask.scatter_(1, min_encoding_ids, 1)
        latent_reps = torch.matmul(min_encodings_mask, self.code_book.weight).view(enc.shape)
        codebook_loss = F.mse_loss(latent_reps.detach(),enc) + self.beta * F.mse_loss(latent_reps , enc.detach())
        latent_reps = enc + (latent_reps-enc).detach()

        e_mean = torch.mean(min_encodings_mask, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        latent_reps = latent_reps.permute(0, 3, 1, 2).contiguous()

        return latent_reps, codebook_loss, perplexity

class Decoder(nn.Module):
    
    def __init__(self, stride = 2):
        super(Decoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode = 'bilinear')
        self.upsample1 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
        self.conv1 = ResidualStack(128, 64, 256,2)
        self.conv2 = ResidualStack(64, 32, 128,3)
        self.conv3 = ResidualStack(32, 16, 64,3)
        self.conv4 = ResidualStack(16, 8, 32,3)
        self.conv5 = nn.Conv2d(8, 3, kernel_size=1,
                      stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,X):
        X = self.conv1(X) 
        X = self.upsample1(X, output_size=(X.size(0),X.size(1),X.size(2)*2,X.size(3)*2))        #56
        X = self.conv2(X) 
        X = self.upsample2(X, output_size=(X.size(0),X.size(1),X.size(2)*2,X.size(3)*2))       #112
        X = self.conv3(X)
        X = self.upsample3(X, output_size=(X.size(0),X.size(1),X.size(2)*2,X.size(3)*2))       #224
        X = self.conv4(X)
        X = self.conv5(X)
        return self.sigmoid(X)


class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, in_dim, kernel_size=1,
                      stride=1, bias=False)
        )
        

    def forward(self, x):
        x = x + self.res_block(x)
        x = F.relu(x)
        return x


class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, res_h_dim)]*n_res_layers)
        self.conv = nn.Conv2d(in_dim, h_dim, kernel_size=1)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(self.conv(x))
        return x
    
class BirdDataset(Dataset):

    def __init__(self, img_pths, device = "cpu"):
        self.device = device
        self.img_pths = img_pths

    def __len__(self):
        return len(self.img_pths)
    
    def __getitem__(self, id):
        img = cv2.imread(self.img_pths[id], cv2.IMREAD_COLOR)
        img_input = img/255.0
        img_input = np.moveaxis(img_input, 2, 0)
        img_input = torch.tensor(img_input, dtype = torch.float).to(self.device)
        return img_input

import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:2].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[2:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
            
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss
    

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum(dim=-1)
        return b.mean()

def print_grads(b):
    for i in b:
        print(i.grad)

class Trainer:
    def __init__(self, encoder, discrete, decoder, optimizer,scheduler,
                 device = DEVICE, vqloss_coeff = 1, commitloss_coeff=0.5, load_pretrained=True):
        self.encoder = encoder
        self.discrete = discrete
        self.decoder = decoder
        self.device = device
        self.mse = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss().to(DEVICE)
        self.ssim = SSIM(data_range=255, size_average=True, channel=3)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.vqloss_coeff = vqloss_coeff
        self.commitloss_coeff = commitloss_coeff
        self.entropy = HLoss()
        self.best_loss = None
        if load_pretrained:
            self.load()
    
    def BCE(self, GT, PRED):
        PRED = torch.clip(PRED, 1e-4, 1-1e-4)
        l = -((1-PRED)**2)*GT*torch.log(PRED) 
        l = torch.sum(l, (1,2,3))
        l = torch.mean(l)
        return l 

    def fit(self, train_loader, l):
        train_loader = tqdm(train_loader)
        train_loader = enumerate(train_loader)
        c_loss = 0
        r_loss = 0
        p = 0
        self.encoder.train()
        self.discrete.train()
        self.decoder.train()
        for i,X in train_loader:
            enc = self.encoder(X)
            latent_reps, codebook_loss, perplexity = self.discrete(enc)
            final_image = self.decoder(latent_reps)
            recon_loss = (self.perceptual_loss(final_image*255.0,X*255.0)  +1 - self.ssim(X*255.0, final_image*255.0)))
            self.optimizer.zero_grad()
            (recon_loss+codebook_loss).backward()
            self.optimizer.step()
            c_loss+=codebook_loss.detach().item()
            r_loss+=recon_loss.detach().item()
            p+=perplexity.detach().item()
        c_loss/=(i+1)
        recon_loss/=(i+1)
        p/=(i+1)
        print("codebook_loss : {} | recon_loss : {} | perplexity : {}".format(c_loss, recon_loss, p))
        self.save(c_loss+recon_loss)
        self.scheduler.step(c_loss+recon_loss)

    def test(self, train_loader):
        for X in train_loader:
            break
        self.encoder.eval()
        self.discrete.eval()
        self.decoder.eval()
        i = random.choice([i for i in range(X[0].size()[0])])
        latent_reps, codebook_loss, perplexity = self.discrete(self.encoder(X))
        output = self.decoder(latent_reps)
        output_1 = output.detach().cpu().numpy()[i]
        X_1 = X.cpu().numpy()[i]
        X_1 = np.transpose(X_1, [2,1,0])
        output_1 = np.transpose(output_1, [2,1,0])
        X_1 = np.asarray(X_1*255.0, dtype = np.int32)
        output_1 = np.asarray(output_1*255.0, dtype = np.int32)
        fig,a =  plt.subplots(2,2,figsize=(10,10))
        a[0][0].imshow(X_1)
        a[0][0].set_xticks([])
        a[0][0].set_yticks([])
        a[0][1].imshow(output_1)
        a[0][1].set_xticks([])
        a[0][1].set_yticks([])
        output_2 = output.detach().cpu().numpy()[i+1]
        X_2 = X.cpu().numpy()[i+1]
        X_2 = np.transpose(X_2, [2,1,0])
        output_2 = np.transpose(output_2, [2,1,0])
        X_2 = np.asarray(X_2*255.0, dtype = np.int32)
        output_2 = np.asarray(output_2*255.0, dtype = np.int32)
        a[1][0].imshow(X_2)
        a[1][0].set_xticks([])
        a[1][0].set_yticks([])
        a[1][1].imshow(output_2)
        a[1][1].set_xticks([])
        a[1][1].set_yticks([])
        plt.savefig("test.png")
        plt.close()

    def save(self,loss):
        if self.best_loss == None:
            self.best_loss = loss
            torch.save(self.encoder.state_dict(), "/home/b170007ec/Programs/VQVAE/models/encoder.pth")
            torch.save(self.discrete.state_dict(), "/home/b170007ec/Programs/VQVAE/models/discrete.pth")
            torch.save(self.decoder.state_dict(), "/home/b170007ec/Programs/VQVAE/models/decoder.pth")
            print("weight saved!!")
            
        elif loss < self.best_loss:
            self.best_loss = loss
            torch.save(self.encoder.state_dict(), "/home/b170007ec/Programs/VQVAE/models/encoder.pth")
            torch.save(self.discrete.state_dict(), "/home/b170007ec/Programs/VQVAE/models/discrete.pth")
            torch.save(self.decoder.state_dict(), "/home/b170007ec/Programs/VQVAE/models/decoder.pth")
            print("weight saved!!")
    
    def load(self):
        print("loading weights...")
        self.encoder.load_state_dict(torch.load( "/home/b170007ec/Programs/VQVAE/models/encoder.pth"))
        self.discrete.load_state_dict(torch.load( "/home/b170007ec/Programs/VQVAE/models/discrete.pth"))
        self.decoder.load_state_dict(torch.load( "/home/b170007ec/Programs/VQVAE/models/decoder.pth"))
        print("weights loaded..")
        
root_pth = "/data1/home/b170007ec/vqvae"
birds285_folders = os.listdir(root_pth + "/285 birds/train")
img_pths = []
for folder in birds285_folders:
    folder_pth = root_pth+"/285 birds/train/"+folder
    files = os.listdir(folder_pth)
    for f in files:
        file_pth = folder_pth + "/" + f
        img_pths.append(file_pth)
        
BD = BirdDataset(img_pths, device = DEVICE)
l = len(BD)
train_dataloader = DataLoader(BD,
                              batch_size = 100,
                              shuffle = True)
encoder = Encoder().to(DEVICE)
discrete = discretize(512, 128).to(DEVICE)
decoder = Decoder().to(DEVICE)
#optimizer = Adam(list(encoder.parameters())+list(decoder.parameters())+list(discrete.parameters()), lr = 0.001)
optimizer = Adam([{'params': list(encoder.parameters())+list(decoder.parameters())},
                {'params': list(discrete.parameters()), 'lr': 1e-1}
            ], lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1,patience=5,verbose=True)
trainer = Trainer(encoder, discrete, decoder, optimizer,scheduler, load_pretrained=False)

trainer.load()
for epoch in range(300):
    print(epoch)
    trainer.test(train_dataloader)
    trainer.fit(train_dataloader, l)