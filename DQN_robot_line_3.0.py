import tkinter as tk  # 图行界面化设计
import torch#用于保存和读取网路模型
import torch.optim as optim #实现了多种优化算法的包
from torch.autograd import Variable #一种可以不断变化的变量，符合反向传播
import torch.nn.functional as F  #非线性激活函数
import torch.nn as nn#包含各种层，方便我们调用构建网络如：全连接层、激活函数层等
import numpy as np#生成数组等用到
import time #时间模块
Batch_size = 35   #从缓冲区采样过程的批量大小
Lr = 0.01          #学习率
Epsilon = 0.91      #epsilon-贪婪概率
Gamma = 0.91        #奖励折扣系数
Target_replace_iter = 200 #对目标更新的频次
Memory_capacity = 6000  #经验池大小
N_actions = 8 #action个数为8，上下左右(0,1,2,3)和左上右上左下右下(4,5,6,7)
N_states = 2#传入网路的参数数量为2，机器人的坐标值(x,y)
ENV_A_SHAPE = 0#确认形状
class Maze(tk.Tk):#创建地图
    UNIT = 50  # 像素大小
    MAZE_H = 9  # 画布长度
    MAZE_W = 9  # 画布宽度
    def __init__(self):#设置地图
        super().__init__()
        self.title('折现D_Q_earning')
        h = self.MAZE_H * self.UNIT#窗口的高
        w = self.MAZE_W * self.UNIT#窗口的宽
        self.canvas = tk.Canvas(self, bg='pink', height=h, width=w)#窗口的颜色设置
        #在可行走路径上设置了7个障碍显示为灰色
        self._draw_rect(0, 6, 'grey47')
        self._draw_rect(1, 0, 'grey47')
        self._draw_rect(1, 3, 'grey47')
        self._draw_rect(4, 1, 'grey47')
        self._draw_rect(7, 0, 'grey47')
        self._draw_rect(7, 7, 'grey47')
        self._draw_rect(8, 4, 'grey47')
        #黄色边界线，向右偏移3像素值利于显示
        line1 = self.canvas.create_line(3, 3, 3, 447, fill='yellow', width=3)
        line2 = self.canvas.create_line(3, 3, 447, 3, fill='yellow', width=3)
        line3 = self.canvas.create_line(447, 3, 447, 447, fill='yellow', width=3)
        line4 = self.canvas.create_line(3, 447, 97, 447, fill='yellow', width=3)
        line5 = self.canvas.create_line(97, 447, 97, 103, fill='yellow', width=3)
        line6 = self.canvas.create_line(97, 103, 347, 103, fill='yellow', width=3)
        line7 = self.canvas.create_line(347, 103, 347, 447, fill='yellow', width=3)
        line8 = self.canvas.create_line(347, 447, 447, 447, fill='yellow', width=3)
        #红色的五角星
        line17 = self.canvas.create_line(230, 120, 210, 160, fill='red', width=3)
        line18 = self.canvas.create_line(230, 120, 250, 160, fill='red', width=3)
        line18 = self.canvas.create_line(203, 134, 250, 160, fill='red', width=3)
        line16 = self.canvas.create_line(203, 134, 257, 134, fill='red', width=3)
        line16 = self.canvas.create_line(257, 134, 210, 160, fill='red', width=3)
        #显示出终点的方位，设置为白色
        self._draw_rect(8, 8, 'white')
        # 显示寻路机器人，初始位置(0,8),设置橘红色
        self.rect = self._draw_rect(0, 8, 'orangered')
        self.canvas.pack()  #显示整个画出的窗口

    def _draw_rect(self, x, y, color):#画矩形，x,y表示横、竖第几个格子(从左上到右下)
        padding = 6  # 内边距6px
        coor = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,
                self.UNIT * (y + 1) - padding]#根据内边距以及每一个格子的坐标得到每一个方块的四个顶点设定
        return self.canvas.create_rectangle(*coor, fill=color)#画出矩形并显示其颜色

    def move_to(self,state, delay=0.5):#根据传入的状态,玩家移动到新位置，每0.5更新位置移动
        coor_old = self.canvas.coords(self.rect)  # 得到机器人修改线的位置，便于更新位置
        x, y = state % 9,state // 9  #横竖第几个格子
        padding = 6  # 内边距6px
        coor_new = [self.UNIT * x + padding, self.UNIT * y + padding, self.UNIT * (x + 1) - padding,self.UNIT * (y + 1) - padding]#获取新的修改线
        dx_pixels, dy_pixels = coor_new[0] - coor_old[0], coor_new[1] - coor_old[1]  # 左上角顶点坐标之差
        self.canvas.move(self.rect, dx_pixels, dy_pixels)#个体位置更新，移动
        self.update() #更新
        time.sleep(delay) #机器人每次移动的速度控制

    def re_location(self):#定义起点以及是否到终点的done值，便于确认是否到达终点
        self.x = 0
        self.y = 8
        self.done = False

    def boundary_limit(self, x, y):
    #纠正坐标值，防止越出边界，首先设置规定在9*9的方格内，然后再次规定区域，使得机器人在黄色线内移动
        x = max(x, 0)
        x = min(x, 8)
        y = max(y, 0)
        y = min(y, 8)
        if 1 < x < 7 and 1 < y:#越过边界退后一步处理
            if y>2 and x==2:
                x=x-1
            elif y>2 and x==6:
                x=x+1
            elif y == 2 and 1<x<7:
                y=y-1
        return x, y


    def Enviroment_interaction(self,action):#动作更新，并得到奖励值
        self.done = False#终点判断
        reward = 0#初始化
        if action==0:#向上移动
            self.x,self.y = self.x,self.y-1
        elif action==1:#向下移动
            self.x,self.y = self.x,self.y+1
        elif action==2:#向左移动
            self.x,self.y = self.x-1,self.y
        elif action==3:#向右移动
            self.x,self.y = self.x+1,self.y
        elif action==4:#向左上移动
            self.x,self.y = self.x-1,self.y-1
        elif action==5:#向右上移动
            self.x,self.y = self.x+1,self.y-1
        elif action==6:#向左下移动
            self.x,self.y = self.x-1,self.y+1
        elif action==7:#向右下移动
            self.x,self.y = self.x+1,self.y+1
        self.x, self.y = self.boundary_limit(self.x, self.y)#得到可执行的下一个状态坐标
        #奖励值的设定，从起点到终点通过设置函数使得奖励值依次上升(总体上来说)，确保有很好的收敛性
        #对于设定的7个陷阱再次修改奖励为 -100，机器人也能有很好的避障功能
        if 1<self.x<7:
            if self.y == 0:
                reward = self.x+2
            elif self.y == 1:
                reward = self.x+1
                if self.x == 4:
                    reward = -100
        else:
            if self.x == 0:
                reward = 1 - 2 * self.y
                if self.y == 6:
                    reward = -100
            elif self.x == 1:
                reward = 3 - 2 * self.y
                if self.y == 3:
                    reward = -100
                elif self.y == 0:
                    reward = -100
            elif self.x == 7:
                reward = 9 + 2 * self.y
                if self.y == 0:
                    reward = -100
                elif self.y == 7:
                    reward = -100
            elif self.x == 8:
                reward = 11 + 2 * self.y
                if self.y == 8:
                    self.done = True
                elif self.y == 4:
                    reward = -100
            else:
                self.done = False
        return np.array((self.x, self.y)), reward, self.done #返回状态坐标以及奖励值和是否到达终点的done值

class Net(nn.Module):  #定义目标网络和训练网络中使用的网络
    def __init__(self):
        super(Net, self).__init__()  # 定义一个简单的全连接网络结构
        self.fc1 = nn.Linear(N_states, 50)  # 定义第一层全连接网络的结构
        self.fc1.weight.data.normal_(0, 0.1)  # 权重初始化为0~0.1之间的随机数
        self.fc2 = nn.Linear(50, 40)  # 定义第二层全连接网络的结构
        self.fc2.weight.data.normal_(0, 0.1)  # 权重初始化为0~0.1之间的随机数
        self.out = nn.Linear(40, N_actions)  # 定义第三层全连接网络的结构
        self.out.weight.data.normal_(0, 0.1)  # 权重初始化为0~0.1之间的随机数

    def forward(self, x):  # 定义输入数据在网络内的传递方式(向前传递)
        x1 = self.fc1(x)  # 起到全连接层分类器的作用,对前层的特征进行一个加权和,将特征空间通过线性变换映射到样本标记空间
        x2 = self.fc2(x1)
        x3 = F.relu(x2)  #当x>=0时，为x，否则为0。
        actions_value = self.out(x3)  # 得出左右0，1行为的动作价值
        return actions_value  # 返回动作价值
class DQN(object):#定义 DQN 网络及其相应的方法
    def __init__(self):
        self.eval_net,self.target_net = Net(),Net()#定义一个训练网络和目标网络
        self.learn_step_counter = 0  #定义计数器，计算学习过程的步骤。
        self.memory_counter = 0  #用于体验重播缓冲区的计数器，存放数据当前状态、动作、奖励和下一状态，而状态值有2(坐标x,y值)
        self.memory = np.zeros((Memory_capacity,N_states*2 + 2))#定义缓冲区，为其分配空间，列数取决于 4 个元素：当前状态、动作、奖励和下一状态，总数为 N_STATES*2 + 2
        self.optimizer = optim.Adam(self.eval_net.parameters(),lr=Lr)#定义优化器，使用已设置的学习率
        self.loss_func = nn.MSELoss()#定义损失函数-均方误差
    def choose_action(self,x):#该函数用于根据 epsilon 贪婪做出决策
        x = Variable(torch.unsqueeze(torch.FloatTensor(x),0))#向输入状态 X 添加 1 个维度
        if np.random.uniform() < Epsilon:#ε贪婪策略下
            action_value = self.eval_net.forward(x)#向前传递，得到动作价值
            action = torch.max(action_value, 1)[1].data.numpy()#返回最大值所相应的索引组成的张量，通过这个索引来代表机器人的动作。
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)#返回 max_action的索引
        else:
            action = np.random.randint(0,N_actions)#随机选取动作
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)#根据条件对随机选取的动作赋值或者对ENV_A_SHAPE数组进行重塑
        return action#返回动作
    def store_transition(self,s,a,r,s_):#此函数充当体验重播缓冲区
        transition = np.hstack((s,[a,r],s_))#水平堆叠这些向量
        index = self.memory_counter % Memory_capacity#如果容量已满，则使用索引将旧内存替换为新内存# 如果记忆库满了, 就覆盖老数据
        self.memory[index,:] = transition
        self.memory_counter += 1
    def learn(self):# 定义整个 DQN 的工作原理，包括何时以及如何更新目标网络的参数和如何实现反向传播。
        if self.learn_step_counter % Target_replace_iter == 0: #每个固定的步骤更新目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())#将eval_net的参数分配给target_net， eval_net负载参数
        self.learn_step_counter += 1
        sample_index = np.random.choice(Memory_capacity,Batch_size)#从缓冲区确定采样批次的索引，从缓冲区中随机选择一些数据
        b_memory = self.memory[sample_index,:]# 根据索引从缓冲区中提取批量大小的经验。
        # 从批处理内存中提取Batch_size个向量或矩阵当前状态、动作、奖励和下一状态并将其转换为便于反向传播的变量
        b_s = Variable(torch.FloatTensor(b_memory[:,:N_states]))
        b_a = Variable(torch.LongTensor(b_memory[:,N_states:N_states+1].astype(int)))# 将长整型转换为张量
        b_r = Variable(torch.FloatTensor(b_memory[:,N_states+1:N_states+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:,-N_states:]))
        #下面的eval_net(b_s)，直接调用eval_net中的forward函数，一次性计算出Batch_size个状态下的行为值。
        q_eval = self.eval_net(b_s).gather(1,b_a)#针对做过的动作b_a, 选择新的q_eval值(q_eval保存所有动作的值)
        q_next = self.target_net(b_s_).detach()#q_next不进行反向传递误差, 所以 detach，计算下一个状态的 q 值
        q_target = b_r +Gamma * q_next.max(1)[0].view(Batch_size, 1)#选择最大的q值，返回沿 axis=1 的最大值及其对应的索引
        loss = self.loss_func(q_eval,q_target)#先计算评估网络和目标网络计算的误差
        self.optimizer.zero_grad()# 然后根据optimizer的设定，进行梯度计算，将梯度重置为零
        loss.backward()#修改网络参数
        self.optimizer.step()# 执行反向传播

def DQN_training():#训练函数
    dqn = DQN()  # 创建DQN类的对象
    env = Maze()#导入环境类
    rewards=[]#用于储存每次的奖励值
    print('\nCollecting experience...')
    for episode in range(2000):
        env.re_location() #定义初始位置
        current_state,done = (np.array([0,8]),False) #得到当前位置的坐标和当前位置的done值
        # env.move_to((current_state[0]+current_state[1]*9))  #更新并显示玩家的位置(将坐标转换为状态值(0~80))
        while not done:#没有到达终点
            reward = 0  # 初始化得分奖励为0
            action = dqn.choose_action(current_state)  # 按一定概率，随机或着贪婪的选择动作
            next_state,next_state_reward,done = env.Enviroment_interaction(action)#与环境进行交互得到下一个状态
            dqn.store_transition(current_state, action, next_state_reward,next_state)  # 保存这一组得分记忆，转换存储状态
            reward += next_state_reward#叠加奖励
            rewards.append(reward)#储存奖励
            # env.move_to((next_state[0]+next_state[1]*9))  #更新并显示玩家的位置(将坐标转换为状态值(0~80))
            if dqn.memory_counter > Memory_capacity:  # 如果体验缓冲区已填充，DQN 将开始学习或更新其参数。
                dqn.learn()  # 经验记忆库满了就开始进行学习
                if done:#到达终点
                    print('第{}学习结束:奖励值={} '.format(episode+1, sum(rewards)))  #记录学习次数和奖励值
                    rewards.clear()#清空
            if done:#到达终点
                rewards.clear()#清空
                break
            current_state = next_state  # 使用下一个状态更新当前状态
    torch.save(dqn.eval_net.state_dict(),'DQN_net11111.pkl')#储存网络模型

def tes_DQN():
    dqn = DQN()  # 创建DQN类的对象
    env = Maze()#导入环境类
    rewards=[]#用于储存每次的奖励值
    print('\ntesting...')
    env.re_location()#定义初始位置
    dqn.eval_net.load_state_dict(torch.load('DQN_net11111.pkl'))#导入训练好的网络模型
    current_state,done = (np.array([0,8]),False)#得到当前位置的坐标和当前位置的done值
    env.move_to((current_state[0] + current_state[1] * 9))  #更新并显示玩家的位置(将坐标转换为状态值(0~80))
    while not done:#没有到达终点
        reward = 0  # 初始化得分奖励为0
        action = dqn.choose_action(current_state)  # 按一定概率，随机或着贪婪选择动作
        next_state,next_state_reward,done = env.Enviroment_interaction(action)#与环境进行交互得到下一个状态
        dqn.store_transition(current_state,action,next_state_reward, next_state)  # 保存这一组得分记忆，转换存储状态
        reward += next_state_reward#叠加奖励
        rewards.append(reward)#储存奖励
        env.move_to((next_state[0] + next_state[1] * 9))  #更新并显示玩家的位置(将坐标转换为状态值(0~80))
        current_state = next_state# 使用下一个状态更新当前状态
    print('测试结束,奖励值={}'.format(sum(rewards)))



if __name__ == '__main__':
    # DQN_training()#训练
    tes_DQN()#测试





