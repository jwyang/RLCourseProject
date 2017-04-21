import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
## load mnist dataset
root = './data'
download = False


# trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Scale(64),
    transforms.ToTensor(),       
    ])

train_set = dset.ImageFolder('data/', transform = trans)
test_set = dset.ImageFolder('data/', transform = trans)
print(train_set.classes)
print(test_set.classes)
#exit()
batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
                 dataset=train_set,
                 batch_size=batch_size,
                 shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
                dataset=test_set,
                batch_size=batch_size,
                shuffle=False, **kwargs)

print '==>>> total trainning batch number: {}'.format(len(train_loader))
print '==>>> total testing batch number: {}'.format(len(test_loader))

## network
class FoodNet(nn.Module):
  def __init__(self):
    super(FoodNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 8, kernel_size=3)
    self.conv1.weight.data.normal_(0, 0.01)
    self.conv1.bias.data.fill_(0)

    self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
    self.conv2.weight.data.normal_(0, 0.01)
    self.conv2.bias.data.fill_(0)

    self.conv3 = nn.Conv2d(8, 8, kernel_size=3)        
    self.conv3.weight.data.normal_(0, 0.01)
    self.conv3.bias.data.fill_(0)

    self.fc1 = nn.Linear(8 * 6 * 6, 8 * 2 * 2)
    self.fc1.weight.data.normal_(0, 0.01)
    self.fc1.bias.data.fill_(0)    
    #self.drop1 = nn.Dropout(0.5)

    self.fc2 = nn.Linear(8 * 2 * 2, 3)
    self.fc2.weight.data.normal_(0, 0.01)
    self.fc2.bias.data.fill_(0)   

    self.bn1 = nn.BatchNorm2d(8)
    self.bn2 = nn.BatchNorm2d(8)
    self.bn3 = nn.BatchNorm2d(8)

    self.ceriation = nn.CrossEntropyLoss()

  def forward(self, x, target):
    x = F.max_pool2d(self.conv1(x), 2)    
    x = self.bn1(x)        
    x = F.relu(x)

    x = F.max_pool2d(self.conv2(x), 2)    
    x = self.bn2(x)        
    x = F.relu(x)

    x = F.max_pool2d(self.conv3(x), 2)    
    x = self.bn3(x)        
    x = F.relu(x)

    x = x.view(-1, 8 * 6 * 6)
    x = self.fc1(x)    
    #x = self.drop1(x)
    x = F.relu(x)
    x = self.fc2(x)
    loss =  self.ceriation(x, target)
    return x, loss
  def name(self):
    return 'foodnet'    
## training
model = FoodNet().cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for epoch in xrange(50):
    # trainning
    for batch_idx, (x, target) in enumerate(train_loader):
        optimizer.zero_grad()
        x, target = Variable(x.cuda()), Variable(target.cuda())
        _, loss = model(x, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print '==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(epoch, batch_idx, loss.data[0])
    # testing
    correct_cnt, ave_loss = 0, 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = Variable(x.cuda(), volatile=True), Variable(target.cuda(), volatile=True)
        score, loss = model(x, target)
        _, pred_label = torch.max(score.data, 1)
        correct_cnt += (pred_label == target.data).sum()
        ave_loss += loss.data[0]
    accuracy = correct_cnt*1.0/len(test_loader)/batch_size
    ave_loss /= len(test_loader)
    print '==>>> epoch: {}, test loss: {:.6f}, accuracy: {:.4f}'.format(epoch, ave_loss, accuracy)
    torch.save(model.state_dict(), model.name() + '_' + str(epoch))
