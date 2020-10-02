import torch
import torch.backends.cudnn as cudnn
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
from torchvision.models import alexnet
root="C:/fruitdb/"
savepath="C:/intelFruit/Alexnet/"
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
        img = opencvLoad(image,227,227)
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
        image = cv2.resize(image, (227, 227), interpolation=cv2.INTER_CUBIC)
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
 
 
#-----------------create the Net and training------------------------
 
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 96, 11, 4, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(96, 256, 5, 1, 2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(256,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(384,384, 3, 1, 1),
            torch.nn.ReLU(),
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(384,256, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,2)
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(9216, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(4096, labelnum )
        )
 
    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)
        res = conv5_out.view(conv5_out.size(0), -1)
        out = self.dense(res)
        return out
 
 
#model = Net()
 
model=Net( )
model_dict=model.state_dict()
pretrained_model=alexnet(pretrained=True)
pretrained_dict=pretrained_model.state_dict()
pretrained_dict={k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict) 
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
    for i in range(labelnum ):
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