import torch
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import datetime  
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F
root="C:/fruitdb/300x300/"
savepath="C:/intelFruit/GOOGLENET/"
classnum=40
#writer = SummaryWriter("C:/intelFruit/sw") 
def saveauc(savefilename,cvnum,i,kk,  auc):
    target = open(savefilename, 'a')
    #s0= "cvnum\trepeatnum\tepcohnum\tauc\n"
    #target.write(s0)    
    s1=str(cvnum)+"\t"+str(i)+"\t"+str(kk)+"\t"+ str(auc)+"\n"
    target.write(s1)
    target.close()   

def savelabelenumdata(savefilename,  probdata):
    
    target = open(savefilename, 'a')
    len1=np.shape(probdata)[0]
    for i in range(len1):
        label=int(probdata[i,0])
        pos="%.0f" % probdata[i,1] 
        neg="%.0f" % probdata[i,2] 
        acc="%.4f" % probdata[i,3] 
        s=str(label) + "\t"+str(pos)+ "\t"+str(neg)+ "\t"+str(acc) +"\n"

        target.write(s)
    target.close()
    
def savecrosspreddata(savefilename, crossdata):
    
    target = open(savefilename, 'a')
    len1=classnum
    len2=classnum
    for i in range(len1): 
        s=""
        
        for j in range(len2): 
            s+= str("%.0f" %crossdata[i,j])+"\t"
        s+="\n"
        target.write(s)
    target.close()   
    
def saveprobdata(savefilename,probdata,test_infor):
    
    target = open(savefilename, 'a')
    
    len1=len(probdata[:,0])
        
    
    for i in range(len1):         
        s=str("%.0f" %probdata[i,0])+"\t"+str("%.0f" %probdata[i,1])+"\t"+  test_infor[i]  
        s+="\n"
        target.write(s)
    target.close()        
    
def savelossaccdata(savefilename,lossdata):
    
    target = open(savefilename, 'a')
    
    len1=len(lossdata[:,0])
    s ="epoch\ttrainloss\trainacc\ttestloss\ttestacc\n"
    target.write(s)        
    
    for i in range(len1):         
        s=str("%.0f" %lossdata[i,0])+"\t"+str("%.7f" %lossdata[i,1]) +"\t"+str("%.7f" %lossdata[i,2]) +"\t"+ str("%.7f" %lossdata[i,3]) +"\t"+str("%.7f" %lossdata[i,4]) 
        s+="\n"
        target.write(s)
    target.close()        
def saveauchead(savefilename ):
    target = open(savefilename, 'a')
    s0= "cvnum\trepeatnum\tepcohnum\tauc\n"
    target.write(s0)    
    
    target.close() 
    
def saveteststr(savefilename,s ):
    target = open(savefilename, 'a')
    
    target.write(s)    
    
    target.close()  

# -----------------ready the dataset--------------------------
def opencvLoad(imgPath,resizeH,resizeW):
    image = cv2.imread(imgPath)
    #print(imgPath)
    image = cv2.resize(image, (resizeH, resizeW), interpolation=cv2.INTER_CUBIC)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 1, 0))  
    image = torch.from_numpy(image)
    return image
    
class LoadPartDataset(Dataset):
    def __init__(self, txt):
        fh = open(txt, 'r')
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            labelList = int(words[0])
            imageList =root+ words[1]
            imgs.append((imageList, labelList))
        self.imgs = imgs
            
    def __getitem__(self, item):
        image, label = self.imgs[item]
        #print(image)
        img = opencvLoad(image,224,224)
        return img,label
    def __len__(self):
        return len(self.imgs)
        
def loadTrainData(txt=None):
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        label = int(words[0])
        image = cv2.imread(root+words[1])
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        image = image.astype(np.float32)
        image = np.transpose(image, (2, 1, 0))  
        image = torch.from_numpy(image)
        imgs.append((image, label))
    return imgs

def loadTestData(txt=None):
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        
        imageList = words[1]
        imgs.append( imageList )
    return imgs
            
 
trainSet =LoadPartDataset(txt=root+'train.txt')
test_data=LoadPartDataset(txt=root+'test.txt')
test_infor=loadTestData(txt=root+'test.txt')
train_loader = DataLoader(dataset=trainSet, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=10)
 


#定义conv-bn-relu函数
def conv_relu(in_channel, out_channel, kernel, stride=1, padding=0):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel, stride, padding),
        nn.BatchNorm2d(out_channel, eps=1e-3),
        nn.ReLU(True),
    )
    return conv

#定义incepion结构，见inception图
class inception(nn.Module):
    def __init__(self, in_channel, out1_1, out2_1, out2_3, out3_1, out3_5,
                 out4_1):
        super(inception, self).__init__()
        self.branch1 = conv_relu(in_channel, out1_1, 1)
        self.branch2 = nn.Sequential(
            conv_relu(in_channel, out2_1, 1),
            conv_relu(out2_1, out2_3, 3, padding=1))
        self.branch3 = nn.Sequential(
            conv_relu(in_channel, out3_1, 1),
            conv_relu(out3_1, out3_5, 5, padding=2))
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            conv_relu(in_channel, out4_1, 1),
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        output = torch.cat([b1, b2, b3, b4], dim=1)
        return output

# 堆叠GOOGLENET，见上表所示结构
class  Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            conv_relu(3, 64, 7, 2, 3), nn.MaxPool2d(3, stride=2, padding=0),
            conv_relu(64, 64, 1), conv_relu(64, 192, 3, padding=1),
            nn.MaxPool2d(3, 2), inception(192, 64, 96, 128, 16, 32, 32),
            inception(256, 128, 128, 192, 32, 96, 64), nn.MaxPool2d(
                3, stride=2), inception(480, 192, 96, 208, 16, 48, 64),
            inception(512, 160, 112, 224, 24, 64, 64),
            inception(512, 128, 128, 256, 24, 64, 64),
            inception(512, 112, 144, 288, 32, 64, 64),
            inception(528, 256, 160, 320, 32, 128, 128), nn.MaxPool2d(3, 2),
            inception(832, 256, 160, 320, 32, 128, 128),
            inception(832, 384, 182, 384, 48, 128, 128), nn.AvgPool2d(2))
        self.classifier = nn.Sequential(
            nn.Linear(9216,1024),
            nn.Dropout2d(p=0.4),
            nn.Linear(1024, classnum))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out
 
model = Net()
 
 
model.cuda()
cudnn.benchmark = True
print(model)
epochnum=200 
loss_acc_mat= np.zeros((epochnum,5),dtype=np.float32)#epoch,train_los, train_acc, eval_loss,val_acc  
 
 
 
 
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam (model.parameters(), lr=0.0001  )
 
model.train() 
for epoch in range(epochnum):
    print('epoch {}'.format(epoch + 1))
    loss_acc_mat[epoch,0]=epoch
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for trainData, trainLabel in train_loader:
        trainData, trainLabel = Variable(trainData.cuda()), Variable(trainLabel.cuda())
        optimizer.zero_grad()
        out = model(trainData)
        loss = loss_func(out, trainLabel)
        #print(loss)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == trainLabel).sum()
        train_acc += train_correct.item()
        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #  if epoch % 100 == 0:
    now_time = datetime.datetime.now()
    now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
    print(now_time,'Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        trainSet)), train_acc / (len(trainSet))))
    #writer.add_scalar('Train/Loss', train_loss/(len(
    #    trainSet)),epoch)
    #writer.add_scalar('Train/Acc',train_acc/(len(
    #    trainSet)),epoch) 
    loss_acc_mat[epoch,1]=    train_loss/(len(
        trainSet))
    loss_acc_mat[epoch,2]=    train_acc/(len(
        trainSet))
    #evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    y_test_com =None# np.zeros((len_test,1),dtype=np.float32)
    y_predicted_com =None #np.zeros((len_test,1),dtype=np.float32)
    pos_test=0
    kk=0
    for testData,testLabel in test_loader:
         testData, testLabel = Variable(testData.cuda()), Variable(testLabel.cuda())
         out = model( testData)
         loss = loss_func(out, testLabel )
         #print(loss)
         eval_loss += loss.item()
         pred = torch.max(out, 1)[1]
         num_correct = (pred == testLabel).sum()
         eval_acc += num_correct.item()


         y_predicted= pred.cpu() .detach().numpy() 
         len1=len( testLabel)
         y_test= testLabel.reshape((len1,1))
         y_predicted=y_predicted.reshape((len1,1))
         y_test= y_test.cpu(). detach().numpy()
   
         if kk==0:
             y_predicted_com =y_predicted
             y_test_com=y_test
         else:
             y_predicted_com =np.vstack((y_predicted_com ,y_predicted))
             y_test_com=  np.vstack((y_test_com ,y_test))             
             
         kk +=1 
 
    #writer.add_scalar('Test/Loss', eval_loss / (len(test_data)),epoch)
    #writer.add_scalar('Test/Acc',eval_acc / (len(test_data)),epoch)    
    loss_acc_mat[epoch,3]=    eval_loss / (len(test_data))
    loss_acc_mat[epoch,4]=    eval_acc  / (len(test_data))
    
    np_data_full=np.hstack((y_test_com ,y_predicted_com ))    
    
    model.zero_grad()
    now_time = datetime.datetime.now()
    now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
    acc=int((eval_acc /  len(test_data))*10000)   
    
    print(now_time,'Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
         test_data)), eval_acc / (len(test_data))))
    ##
    len1=len(y_test_com)
    pred_ct = np.zeros((classnum,4),dtype=np.float32)
    pred_cross = np.zeros((classnum,classnum),dtype=np.float32)
    tt=0
    for i in range(2):
        pred_ct[i,0] =tt
        tt+=1
    
    
    for i in range(len1):
        label=int(y_test_com[i])
        pred=int(y_predicted_com[i])
        pred_cross [label,pred] = pred_cross[label,pred]+1        
        if pred==label:
            pred_ct[label,1] =pred_ct[label,1]+1

        else:
            pred_ct[label,2] =pred_ct[label,2]+1
            
           
    pred_ct[:,3]   =pred_ct[:,1]/(pred_ct[:,1]+pred_ct[:,2])

    savelabelenumdata(savepath+str( acc)+"_pred_"+str(epoch)+".txt", pred_ct)    
    savecrosspreddata(savepath+str( acc)+"_cross_"+str(epoch)+".txt", pred_cross)  
    saveprobdata(savepath+str( acc)+"_prob_"+str(epoch)+".txt", np_data_full,test_infor)  
   
    model.train()
    model.zero_grad()    
#writer.close()  
savelossaccdata(savepath +"lossacc.txt", loss_acc_mat)                