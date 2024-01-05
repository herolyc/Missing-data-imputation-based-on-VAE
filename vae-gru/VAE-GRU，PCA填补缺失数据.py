import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
import torch.utils.data as Data
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_0_1_array(array,rate=0.1):
    '''按照数组模板生成对应的 0-1 矩阵，默认rate=0.2'''
    zeros_num = int(array.size * rate)#根据0的比率来得到 0的个数
    new_array = np.ones(array.size)#生成与原来模板相同的矩阵，全为1
    new_array[:zeros_num] = 0 #将一部分换为0
    np.random.shuffle(new_array)#将0和1的顺序打乱
    re_array = new_array.reshape(array.shape)#重新定义矩阵的维度，与模板相同
    return re_array

def inverse(scaler, y, n_col):
    y = y.copy()
    y -= scaler.min_[n_col]
    y /= scaler.scale_[n_col]
    return y

def seed_torch(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_deb():
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





class gruvae2(nn.Module):
    def __init__(self, step):
        super(gruvae2, self).__init__()

        self.step = step
        self.lr = nn.LeakyReLU(0.5)
        self.sig = nn.Sigmoid()

        self.gru1 = nn.GRU(12, 20, batch_first=True)
        self.line11 = nn.Linear(20, 20)
        self.line12 = nn.Linear(20, 12)

        self.gru2 = nn.GRU(6, 16, batch_first=True)
        self.line2 = nn.Linear(16, 12)

        self.fc1 = nn.Linear(12, 12)

    def forward(self, x):

        batchsz = x.shape[0]
        encoder1, _ = self.gru1(x)
        encoder1 = self.line11(encoder1)
        encoder1 = self.lr(encoder1)
        encoder1 = self.line12(encoder1)
        encoder1 = self.lr(encoder1)
        encoder1 = encoder1 + x

        h_ = encoder1[:, -1, :]
        mu, sigma = h_.chunk(2, dim=1)
        h = mu + sigma * torch.randn_like(sigma)
        h = change_shape(h.detach().numpy(), self.step)
        h = Variable(torch.from_numpy(h)).float()

        decoder2, _ = self.gru2(h)
        decoder2 = self.line2(decoder2)
        decoder2 = self.lr(decoder2)
        decoder2 = decoder2 + encoder1[self.step:,:,:]

        out = self.fc1(decoder2)
        out = self.sig(out)
        #out = out[:, -1, :]

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / batchsz

        return out, kld

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
        encoder1 = encoder1 + x


        h_ = encoder1[:, -1, :]
        mu, sigma = h_.chunk(2, dim=1)
        h = mu + sigma * torch.randn_like(sigma)
        h = change_shape(h.detach().numpy(), self.step)#二维变三维
        h = Variable(torch.from_numpy(h)).float()

        decoder2, _ = self.gru2(h)
        decoder2 = self.line2(decoder2)
        decoder2 = self.lr(decoder2)
        #decoder2 = decoder2 + encoder1[self.step:,:,:]

        out,_ = self.gru3(h)
        out = out + encoder1[self.step:,:,:]
        out=self.line3(out)
        out = self.lr(out)
        out = out[:, -1, :]

        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / batchsz

        return out, kld,decoder2

def frame1(vae, data01, data10, train, nstep):

    train2 = train[:, -1, :].detach().numpy()
    train2 = train2[nstep:, :]
    data01 = data01[nstep * 2:, :]
    data10 = data10[nstep * 2:, :]
    train1, _ = vae(train)
    train1 = train1[:,-1,:].detach().numpy()
    train = train1 * data01 + train2 * data10
    train = change_shape(train, nstep)
    train = Variable(torch.from_numpy(train)).float()

    return train ,train1

def filp1(a):
    a_int = a.astype(int)

    # 对数组进行逻辑取反操作
    a_int_not = np.logical_not(a_int)

    # 将结果数组中的元素转换回浮点型
    b = a_int_not.astype(float)
    return b

def avg(arr):
    zero_indices = np.argwhere(arr == 0)

    # 遍历每个 0 元素所在的列，并将其替换为该列的平均值
    for idx in zero_indices:
        row_idx, col_idx = idx
        col = arr[:, col_idx]
        col_mean = np.mean(col[col != 0])  # 排除 0 元素的平均值
        arr[row_idx, col_idx] = col_mean

    return arr

def before(arr):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] == 0:
                if i > 0:
                    arr[i, j] = arr[i - 1, j]  # 使用该列的前一个非 0 元素
                else:
                    arr[i, j] = arr[i, j - 1]  # 如果该列的第一个元素为 0，则使用该行的前一个元素

    return arr

if __name__ == "__main__":
        seed_torch(42)

        nstep = 3

        data = load_deb()
        scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据归一到0到1，可以根据数据特点归一到-1到1
        data = scaler.fit_transform(data)  # 归一化
        num = int(data.shape[0] * 0.5)
        train_data = data[:num, :-1]
        test_data = data[num:, :]

        datax = test_data[:,:-1]
        data_10 = get_0_1_array(datax,0.1)
        data_30 = get_0_1_array(datax, 0.3)
        data_50 = get_0_1_array(datax, 0.5)
        data_10_ = filp1(data_10)
        data_30_ = filp1(data_30)
        data_50_ = filp1(data_50)


        datax10 = np.multiply(data_10, datax)
        datax30 = np.multiply(data_30, datax)
        datax50 = np.multiply(data_50, datax)

        print(datax10)
        datax10 = before(datax10)
        datax30 = before(datax30)
        datax50=before(datax50)

        train_data_3d = change_shape(train_data, nstep)
        datax10_3d =change_shape(datax10 , nstep)
        datax30_3d = change_shape(datax30, nstep)
        datax50_3d = change_shape(datax50, nstep)
        datax10_torch =Variable(torch.from_numpy(datax10_3d)).float()
        datax30_torch = Variable(torch.from_numpy(datax30_3d)).float()
        datax50_torch = Variable(torch.from_numpy(datax50_3d)).float()

        train_data_torch =Variable(torch.from_numpy(train_data_3d)).float()

        newvae = gruvae2(nstep)
        opt_model = torch.optim.Adam(newvae.parameters(), lr=0.1)
        for epoch in range(1000):

            opt_model.zero_grad()
            out, kld = newvae(train_data_torch)
            loss_mse = F.mse_loss(out, train_data_torch[nstep:,:,:])

            loss = loss_mse + 0.001 * kld
            loss.backward()

            opt_model.step()

            if epoch % 100 == 0:
                print(epoch)
                print(loss_mse)
                print(kld)

        model1 = torch.load("gruvae1.pth")

        new_data10, data_imputation = frame1(newvae, data_10_, data_10, datax10_torch, nstep)
        print("datax10:\n", datax)

        new_data10 = new_data10[:, -1, :].detach().numpy()
        print("data_imputation:\n", new_data10)
        # new_data10 = np.hstack((new_data10, data[-(new_data10.shape[0]):, -1].reshape(-1, 1)))
        new_data10_3d = change_shape(new_data10, nstep)
        newdata103d_torch = Variable(torch.from_numpy(new_data10_3d)).float()
        # print(newdata103d_torch.shape)
        out10, _, _ = model1(newdata103d_torch)
        ytest = test_data[-(out10.shape[0]):, -1]
        out10 = inverse(scaler, out10.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(out10, ytest))
        mae = mean_absolute_error(out10, ytest)
        r2 = r2_score(out10, ytest)
        print('gruvae')
        print(rmse, r2, mae)
        print(out10.shape)
        print(ytest.shape)
        x1 = np.arange(0, out10.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, out10, color='blue', linestyle="--", linewidth=1, label="VAE-GRU(missing-10%)")
        plt.legend()
        plt.show()

        new_data30, _ = frame1(newvae, data_30_, data_30, datax30_torch, nstep)
        new_data30 = new_data30[:, -1, :].detach().numpy()
        # new_data10 = np.hstack((new_data10, data[-(new_data10.shape[0]):, -1].reshape(-1, 1)))
        new_data30_3d = change_shape(new_data30, nstep)
        newdata303d_torch = Variable(torch.from_numpy(new_data30_3d)).float()
        # print(newdata103d_torch.shape)
        out30, _, _ = model1(newdata303d_torch)
        ytest = test_data[-(out30.shape[0]):, -1]
        out30 = inverse(scaler, out30.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(out30, ytest))
        mae = mean_absolute_error(out30, ytest)
        r2 = r2_score(out30, ytest)
        print('gruvae')
        print(rmse, r2, mae)
        x1 = np.arange(0, out30.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, out30, color='blue', linestyle="--", linewidth=1, label="VAE-GRU(missing-30%)")
        plt.legend()
        plt.show()

        pca = PCA(n_components=0.5)
        pca.fit(datax10)
        x_pca = pca.transform(datax10)
        data10_pca = pca.inverse_transform(x_pca)
        # print(data10_pca)
        data10_pca = data10_pca * data_10_ + datax10
        print("pca:\n", )
        # data10_pca= data10_pca[:, -1, :].detach().numpy()
        # new_data10 = np.hstack((new_data10, data[-(new_data10.shape[0]):, -1].reshape(-1, 1)))
        data10_pca_3d = change_shape(data10_pca, nstep)
        data10_pca_3d_torch = Variable(torch.from_numpy(data10_pca_3d)).float()
        # print(newdata103d_torch.shape)

        out_pca, _, _ = model1(data10_pca_3d_torch)
        ytest = test_data[-(out_pca.shape[0]):, -1]
        out_pca = inverse(scaler, out_pca.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(out_pca, ytest))
        mae = mean_absolute_error(out_pca, ytest)
        r2 = r2_score(out_pca, ytest)
        print('pca')
        print(rmse, r2, mae)
        x1 = np.arange(0, out_pca.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, out_pca, color='blue', linestyle="--", linewidth=1, label="PCA(missing-10%)")
        plt.legend()
        plt.show()

        pca = PCA(n_components=0.5)
        pca.fit(datax30)
        x_pca = pca.transform(datax30)
        data30_pca = pca.inverse_transform(x_pca)
        # print(data10_pca)
        data30_pca = data30_pca * data_30_ + datax30
        print("pca:\n", )
        # data10_pca= data10_pca[:, -1, :].detach().numpy()
        # new_data10 = np.hstack((new_data10, data[-(new_data10.shape[0]):, -1].reshape(-1, 1)))
        data30_pca_3d = change_shape(data30_pca, nstep)
        data30_pca_3d_torch = Variable(torch.from_numpy(data30_pca_3d)).float()
        # print(newdata103d_torch.shape)

        out_pca, _, _ = model1(data30_pca_3d_torch)
        ytest = test_data[-(out_pca.shape[0]):, -1]
        out_pca = inverse(scaler, out_pca.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(out_pca, ytest))
        mae = mean_absolute_error(out_pca, ytest)
        r2 = r2_score(out_pca, ytest)
        print('pca')
        print(rmse, r2, mae)
        x1 = np.arange(0, out_pca.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, out_pca, color='blue', linestyle="--", linewidth=1, label="PCA(missing-30%)")
        plt.legend()
        plt.show()

        s1 = np.zeros([datax10.shape[0], 3])
        u, s, v = np.linalg.svd(datax10)
        for i in range(3):
            s1[i, i] = s[i]
        v = v[:3, :]
        data10_svd = np.dot(np.dot(u, s1), v)
        data10_svd = data10_svd * data_10_ + datax10

        data10_svd_3d = change_shape(data10_svd, nstep)
        data10_svd_3d_torch = Variable(torch.from_numpy(data10_svd_3d)).float()
        # print(newdata103d_torch.shape)

        svd_10, _, _ = model1(data10_svd_3d_torch)
        ytest = test_data[-(svd_10.shape[0]):, -1]
        svd_10 = inverse(scaler, svd_10.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(svd_10, ytest))
        mae = mean_absolute_error(svd_10, ytest)
        r2 = r2_score(svd_10, ytest)
        print('svd')
        print(rmse, r2, mae)
        x1 = np.arange(0, svd_10.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, svd_10, color='blue', linestyle="--", linewidth=1, label="SVD(missing-10%)")
        plt.legend()
        plt.show()

        s1 = np.zeros([datax30.shape[0], 3])
        u, s, v = np.linalg.svd(datax30)
        for i in range(3):
            s1[i, i] = s[i]
        v = v[:3, :]
        data30_svd = np.dot(np.dot(u, s1), v)
        data30_svd = data30_svd * data_30_ + datax30

        data30_svd_3d = change_shape(data30_svd, nstep)
        data30_svd_3d_torch = Variable(torch.from_numpy(data30_svd_3d)).float()
        # print(newdata103d_torch.shape)

        svd_30, _, _ = model1(data30_svd_3d_torch)
        ytest = test_data[-(svd_30.shape[0]):, -1]
        svd_30 = inverse(scaler, svd_30.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(svd_30, ytest))
        mae = mean_absolute_error(svd_30, ytest)
        r2 = r2_score(svd_30, ytest)
        print('svd')
        print(rmse, r2, mae)
        x1 = np.arange(0, svd_30.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, svd_30, color='blue', linestyle="--", linewidth=1, label="SVD(missing-30%)")
        plt.legend()
        plt.show()

        data_mean_s = np.multiply(data_10, datax)
        mean_values =np.nanmean(data_mean_s,axis=0)
        inds =np.where(np.isnan(datax10))
        data_mean_s[inds]=np.take(mean_values,inds[1])
        data_mean = data_mean_s
        data_mean_3d = change_shape(data_mean, nstep)
        data_mean_3d_torch = Variable(torch.from_numpy(data_mean_3d)).float()
        # print(newdata103d_torch.shape)

        out_mean, _, _ = model1(data_mean_3d_torch)
        ytest = test_data[-(out_mean.shape[0]):, -1]
        out_mean = inverse(scaler, out_mean.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(out_mean, ytest))
        mae = mean_absolute_error(out_mean, ytest)
        r2 = r2_score(out_mean, ytest)
        print('mean')
        print(rmse, r2, mae)
        x1 = np.arange(0, out_mean.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, out_mean, color='blue', linestyle="--", linewidth=1, label="mean(missing-10%)")
        plt.legend()
        plt.show()

        data_mean_s = np.multiply(data_30, datax)
        mean_values =np.nanmean(data_mean_s,axis=0)
        inds =np.where(np.isnan(datax30))
        data_mean_s[inds]=np.take(mean_values,inds[1])
        data_mean = data_mean_s
        data_mean_3d = change_shape(data_mean, nstep)
        data_mean_3d_torch = Variable(torch.from_numpy(data_mean_3d)).float()
        # print(newdata103d_torch.shape)

        out_mean, _, _ = model1(data_mean_3d_torch)
        ytest = test_data[-(out_mean.shape[0]):, -1]
        out_mean = inverse(scaler, out_mean.detach().numpy(), -1).reshape((-1, 1))
        ytest = inverse(scaler, ytest, -1).reshape((-1, 1))
        rmse = np.sqrt(mean_squared_error(out_mean, ytest))
        mae = mean_absolute_error(out_mean, ytest)
        r2 = r2_score(out_mean, ytest)
        print('mean')
        print(rmse, r2, mae)
        x1 = np.arange(0, out_mean.shape[0])
        plt.figure(1)
        plt.plot(x1, ytest, color='red', linewidth=1.2, label="Actual Value")
        plt.plot(x1, out_mean, color='blue', linestyle="--", linewidth=1, label="mean(missing-30%)")
        plt.legend()
        plt.show()