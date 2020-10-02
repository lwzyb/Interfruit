import torch
import torch.backends.cudnn as cudnn
import cv2
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import datetime  
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F
root="C:/fruitdb/300x300/"
savepath="C:/intelFruit/vgg16/"
labelnum=40
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
    len1=labelnum 
    len2=labelnum 
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
train_loader = DataLoader(dataset=trainSet, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=4)
 
 
#-----------------create the Net and training------------------------
 
# 用于ResNet18和34的残差块，用的是2个3x3的卷积
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=labelnum):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(25088, num_classes)#512*block.expansion

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        #print(np.shape( out ))
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet50()
    y = net(torch.randn(1,3,32,32))
    print(y.size())
 
 
model = ResNet18()
 
 
 
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
    pred_ct = np.zeros((labelnum ,4),dtype=np.float32)
    pred_cross = np.zeros((labelnum ,labelnum ),dtype=np.float32)
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