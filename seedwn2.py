# 这是一堆初始化
import gym
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# env = gym.make('CartPole-v0')
env = gym.make('MountainCar-v0')  # action = (0,1,2) = (left, no_act, right)
# env = gym.make('Hopper-v3')
print(env.observation_space)


# print(env.action_space)

# 简单的线性模型
def GetModel():
    # In features:2(state) ,out:3 action q
    return nn.Sequential(nn.Linear(2, 16),
                         nn.LeakyReLU(inplace=True),
                         nn.Linear(16, 24),
                         nn.LeakyReLU(inplace=True),

                         nn.Linear(24, 3))


# 创建数据集
class RLDataset(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        # samples = [(s,a,r,s_), ...]
        self.samples = self.transform(samples)

    def __getitem__(self, index):
        # if self.transform is not None:
        #    img = self.transform(img)
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def transform(self, samples):
        transSamples = []
        for (s, a, r, s_) in samples:
            sT = torch.tensor(s, ).float()
            sT_ = torch.tensor(s_).float()
            transSamples.append((sT, a, r, sT_))
        return transSamples


# 采样环境函数，可以设置随机操作的概率。重点在于reward的设计
def GetSamplesFromEnv(env, model, epoch, max_steps, drop_ratio=0.8):
    train_samples = []
    each_sample = None
    env.reset()
    observation_new = None
    observation_old = None
    model.eval()
    for i_episode in range(epoch):
        observation_new = env.reset()
        observation_old = env.reset()
        for t in range(max_steps):
            # env.render()
            # print(observation)
            if random.random() > 1 - drop_ratio:
                action = env.action_space.sample()
            else:
                inputT = torch.tensor(observation_new).float()
                action = torch.argmax(model(inputT)).item()
                # print(action)
            observation_new, reward, done, info = env.step(action)
            # print(reward)
            # We record samples.
            if t > 0:
                # reward += observation_new[0]
                # if observation_new[0] > -0.35:
                #    reward += (observation_new[0] + 0.36)*5
                if observation_new[0] > -0.2:
                    reward += 0.2
                elif observation_new[0] > -0.15:
                    reward += 0.5
                elif observation_new[0] > -0.1:
                    reward += 0.7
                each_sample = (observation_old, action, reward, observation_new)
                train_samples.append(each_sample)

            observation_old = observation_new

            if done:
                # 失败的采样不打印出来
                if t != 199:
                    if t<90:
                        print("Episode finished after {} steps".format(t + 1))
                    #print("Episode finished after {} timesteps".format(t + 1))
                break
    return train_samples
# 训练网络。这里可能gather函数比较绕，还有双网络更新比较费解。忽略掉这些，和正常训练循环一样
# gamma是贝尔曼方程里的衰减因子
def TrainNet(net_target, net_eval, trainloader, criterion, optimizer, device, epoch_total, gamma):
    running_loss = 0.0
    iter_times = 0
    net_target.eval()
    net_eval.train()
    for epoch in range(epoch_total + 1):
        if epoch > 0:
            print('epoch %d, loss %.5f' % (epoch, running_loss))
        running_loss = 0.0
        if epoch == epoch_total:
            break
        for i, data in enumerate(trainloader, 0):
            if iter_times % 100 == 0:
                net_target.load_state_dict(net_eval.state_dict())
            s, a, r, s_ = data
            optimizer.zero_grad()

            # output = Q_predicted.
            q_t0 = net_eval(s)
            q_t1 = net_target(s_).detach()
            q_t1 = gamma * (r + torch.max(q_t1, dim=1)[0])

            loss = criterion(q_t1.float(), torch.gather(q_t0, dim=1, index=a.unsqueeze(1)).squeeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iter_times += 1
    net_target.load_state_dict(net_eval.state_dict())
    print('Finished Training')


# 最后是一大堆主循环
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_target, net_eval = GetModel(), GetModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net_eval.parameters(), lr=0.01)
train_samples = []
goodmodel_idx1 = 1
goodmodel_idx2 = 0
'''
#PATH = '../pythonProject/10.pth'
net_target, net_eval = GetModel(), GetModel()
net_eval.load_state_dict(torch.load(PATH))
net_target.load_state_dict(torch.load(PATH))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net_eval.parameters(), lr=0.01)
train_samples = []


goodmodel_idx = 0
PATH ='56.pth'
net_eval.load_state_dict(torch.load(PATH))
net_target.load_state_dict(torch.load(PATH))
GetSamplesFromEnv(env,net_eval, 20, 200, 0)
#getonestep(env,net_eval,  200, 0)
# 这一堆是测试看效果用的

for i in range(300):
    print(str(goodmodel_idx))
    #PATH = '../pythonProject/pythonProjectttt'+str(goodmodel_idx)+'.pth'
    PATH ='56.pth'
    net_eval.load_state_dict(torch.load(PATH))
    net_target.load_state_dict(torch.load(PATH))
    GetSamplesFromEnv(env,net_eval, 20, 200, 0)

    goodmodel_idx += 1


for t in range(1):
    PATH = '../pythonProject/' + str(goodmodel_idx1) + '.pth'
    net_eval.load_state_dict(torch.load(PATH))
    net_target.load_state_dict(torch.load(PATH))
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net_eval.parameters(), lr=0.01)
    train_samples = []
    goodmodel_idx1=goodmodel_idx1+1
'''
for i in range(300):
        drop_ratio = 0.8 - 0.0077 * i
        sample_times = 10
        tmpSample = GetSamplesFromEnv(env, net_eval, sample_times, 200, drop_ratio)

        train_samples += tmpSample
        # 每次sample的长度就代表了采取的步数，登山车里是越小越好。如果是倒立摆，则是越大越好
        

        if len(tmpSample) < sample_times * 1000:
            print("good model!save it!-------------------------------------------------------------")
            torch.save(net_eval.state_dict(), "goodmodel" + str(goodmodel_idx2) + ".pth")
            goodmodel_idx2 += 1
        
        
        # dataset里存着最新的不超过4000的样本
        if len(train_samples) > 4000:
            train_samples = train_samples[len(tmpSample):len(train_samples)]
        trainset = RLDataset(train_samples)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

        print("battle:",i)
        TrainNet(net_target, net_eval, trainloader, criterion, optimizer, device, 10, 0.9)
        PATH = "pythonProject" + str(i) + ".pth"
        torch.save(net_eval.state_dict(), PATH)


env.close()
