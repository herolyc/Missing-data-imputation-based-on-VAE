from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


def inverse(scaler, y, n_col):#反归一化
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y

def seed_torch(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_deb():#转换数据换为12维
    data = np.genfromtxt("debutanizer_data.txt", delimiter="  ").astype(np.float32)
    x_temp = data[:, :7]
    y_temp = data[:, 7]

    x_new = np.zeros([2390, 12])
    x_6 = x_temp[:, 4]
    x_9 = (x_temp[:, 5] + x_temp[:, 6]) / 2
    x_new[:, :5] = x_temp[4: 2394, :5]
    x_new[:, 5] = x_6[3: 2393]
    x_new[:, 6] = x_6[2: 2392]
    x_new[:, 7] = x_6[1: 2391]
    x_new[:, 8] = x_9[4: 2394]

    x_new[:, 9] = y_temp[3: 2393]
    x_new[:, 10] = y_temp[2: 2392]
    x_new[:, 11] = y_temp[1:2391]
    #x_new[:, 12] = y_temp[:2390]
    y_new = y_temp[4: 2394]
    y_new = y_new.reshape([-1, 1])
    data = np.hstack((x_new, y_new))

    return data

def change_input_shape(data, n_step):
    Y = data[n_step:, -1]
    Y = Y.reshape((Y.shape[0],1))
    X = np.zeros((Y.shape[0], n_step, data.shape[1] - 1))
    for i in range(0, Y.shape[0]):
        X[i] = data[i:i + n_step , : -1]
    return X, Y

def change_shape(data, n_step):
    X = np.zeros((data.shape[0] - n_step, n_step, data.shape[1]))
    for i in range(data.shape[0] - n_step):
        X[i] = data[i:i + n_step , : ]
    return X


class gruvae(nn.Module):
    def __init__(self, step):
        super(gruvae, self).__init__()

        self.step = step
        self.lr = nn.LeakyReLU(0.5)
        #self.sig = nn.Sigmoid()

        self.gru1 = nn.GRU(12, 20, batch_first=True)
        self.line11 = nn.Linear(20, 12)

        self.gru2 = nn.GRU(6, 10, batch_first=True)
        self.line2 = nn.Linear(10, 12)

        self.gru3 = nn.GRU(6, 12, batch_first=True)
        self.line3 = nn.Linear(12, 1)



    def forward(self, x):

        batchsz = x.shape[0]
        encoder1, _ = self.gru1(x)
        encoder1 = self.line11(encoder1)
        encoder1 = self.lr(encoder1)
        encoder1 = encoder1 + x#残差链接


        h_ = encoder1[:, -1, :]
        mu, sigma = h_.chunk(2, dim=1)
        h = mu + sigma * torch.randn_like(sigma)
        h = change_shape(h.detach().numpy(), self.step)#二维变三维
        h = Variable(torch.from_numpy(h)).float()#转换为torch能用的数据

        decoder2, _ = self.gru2(h)
        decoder2 = self.line2(decoder2)
        decoder2 = self.lr(decoder2)#vae输出
        #decoder2 = decoder2 + encoder1[self.step:,:,:]

        out,_ = self.gru3(h)
        out = out + encoder1[self.step:,:,:]
        out=self.line3(out)
        out = self.lr(out)
        out = out[:, -1, :]#预测值

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / batchsz#kl散度

        return out, kld,decoder2

def onlygruvae(xtrain, ytrain, xtest, ytest, scaler, nstep):

    newvae = gruvae(nstep)
    opt_model = torch.optim.Adam(newvae.parameters(), lr=0.01)
    for epoch in range(1000):

        opt_model.zero_grad()
        out, kld,decoder = newvae(xtrain)
        loss_mse = F.mse_loss(out, ytrain)
        loss_x=F.mse_loss(decoder,xtrain[nstep:,:])
        loss = loss_mse + 0.01 * kld+0.01*loss_x
        loss.backward()

        opt_model.step()

        if epoch % 100 == 0:
            print(epoch)
            print(loss_mse)
            print(kld)

    my_out, _,_ = newvae(xtest)
    my_out = my_out.detach().numpy()[:, -1]
    out = inverse(scaler, my_out, -1).reshape((-1, 1))#反归一化
    #print(out.shape)
    #print(ytest.shape)
    rmse = np.sqrt(mean_squared_error(out, ytest))
    mae = mean_absolute_error(out, ytest)
    r2 = r2_score(out, ytest)
    print('gruvae')
    print(rmse, r2, mae)
    return out, newvae


if __name__ == "__main__":
        seed_torch(42)

        nstep = 3

        data = load_deb()

        scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
        data = scaler.fit_transform(data)  # 归一化
        #data = scaler.inverse_transform()

        avg = np.average(data[:,-1])

        print(data.shape)
        num = int(data.shape[0] * 0.7)
        train_data = data[:num, :]
        print(train_data.shape)
        test_data = data[num:, :]
        ytest = scaler.inverse_transform(test_data)[:, -1].reshape((test_data.shape[0], -1))

        train_3d, train_y = change_input_shape(train_data, nstep)
        test_3d, test_y = change_input_shape(test_data, nstep)


        train_3d_torch = Variable(torch.from_numpy(train_3d)).float()
        test_3d_torch = Variable(torch.from_numpy(test_3d)).float()
        train_y_torch = Variable(torch.from_numpy(train_y)).float()
        test_y_torch = Variable(torch.from_numpy(test_y)).float()


        gruvaeout, model = onlygruvae(train_3d_torch, train_y_torch[nstep:, :], test_3d_torch, ytest[nstep * 2:, :], scaler, nstep)

        torch.save(model,"gruvae1.pth")

        x1 = np.arange(0, gruvaeout.shape[0])
        ytest = inverse(scaler, test_y_torch.detach().numpy(), -1)
        plt.figure(1)
        plt.plot(x1, ytest[nstep:, :], color='red', linewidth=1.5, label="Actual Value")
        plt.plot(x1, gruvaeout, color='blue', linestyle="--", linewidth=1.5, label="VAE-GRU")
        plt.legend()
        plt.show()