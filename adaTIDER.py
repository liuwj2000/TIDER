# coding=gbk 
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
from toolz.curried import *
from torch import Tensor
from typing import Optional
import argparse
from sklearn import metrics
import time
import random

warnings.filterwarnings("ignore")

def get_top_freq(x,topk_freq):
    x=torch.Tensor(x)
    mask_x=torch.isnan(x)
    mean_x=torch.nansum(x,dim=1)/torch.sum(mask_x==0,dim=1)
    x_new=x[np.setdiff1d(np.arange(x.shape[0]),np.where(np.isnan(mean_x))),:]
    
    mask_new=torch.isnan(x_new)
    mean_new=torch.nansum(x_new,dim=1)/torch.sum(mask_new==0,dim=1)
    
    for i in range(x_new.shape[0]):
      x_new[i,:].masked_fill_(mask_new[i,:]==True,mean_new[i])

    af=torch.fft.rfft(x_new,dim=1)
    
    freq_A=abs(af).mean(0)
    
    freq_A[0]=0
    aptitude,freq=torch.topk(freq_A,topk_freq)
    
    T=x.shape[1]/freq
    
    T=np.array(list(set(np.int32(T).tolist())))
    
    return T

def evaluated_message(y_test: np.ndarray, y_pred: np.ndarray) -> str:
    rmse = metrics.mean_squared_error(y_test, y_pred, squared=False)
    mae  = metrics.mean_absolute_error(y_test, y_pred)
    msk_0=[]
    for i in range(len(y_pred)):
          if y_test[i]!=0:
            msk_0.append(i)
    y_test, y_pred = y_test[msk_0], y_pred[msk_0]
    abs_errors = np.abs(y_test - y_pred)
    mape = np.mean(abs_errors / y_test)
    msg = (f" RMSE: {rmse:7.4} MAE: {mae:7.4}"
           f" Max Error: {np.max(abs_errors):7.4} MAEP: {mape:7.4}")
    return msg



class TIDER(nn.Module):
    def __init__(self, n: int, t: int, hid_size: int,
                 bias_lag_list, bias_dim_, season_num_,
                 seasonality_,topk_freq,u_bias_dim,n_test):
        super(TIDER, self).__init__()
        self.s_embeddings = nn.Embedding(n, hid_size)
        # channel-wise matrix
        self.t_embeddings_trend = nn.Embedding(t, hid_size)
        # trend feature matrix
        self.t_embeddings_season = nn.Embedding(2*season_num_*topk_freq, hid_size)
        # seasonality feature matrix
        self.u_bias_dim=u_bias_dim
        self.s_embeddings_bias=nn.Embedding(n,u_bias_dim)
        # channel-wise bias matrix
        
        self.topk_freq=topk_freq
        #Topk frequencies
        self.bias_dim = bias_dim_
        # Db
        self.bias_t = nn.Embedding(t, self.bias_dim)
        # bias feature matrix
        self.hidden_size = hid_size
        # dimension for feature matrix
        self.seasonal = seasonality_
        # seasonality
        self.season_num = season_num_
        # degree of Fourier Series

        self.dims = (n, t)
        # num of channels, temporal length
        
        self.param=torch.nn.Parameter(torch.zeros(2))
        # adaptive weight for trend and seasonality
        
        self.modlst = torch.nn.ModuleList([])
        # Autoregressive Linears for bias feature matrix
        self.lag_list = bias_lag_list
        # Lag list for bias matrix
        self.len_lag = len(bias_lag_list)

        for _ in range(self.len_lag):
            self.modlst.append(torch.nn.Linear(self.bias_dim, self.bias_dim, bias=False))
        # one autoregressive matrix corresponds to one time lag
        
        self.sinx=torch.zeros((season_num_,topk_freq,t))
        self.cosx=torch.zeros((season_num_,topk_freq,t))
        
        times=torch.arange(t)
        for i in range(season_num_):
          for j in range(topk_freq):
            # vector sin(ωt) and cos(ωt) for seasonality feature matrix
            self.sinx[i,j,:] = torch.sin((i*2*np.pi/self.seasonal[j])*times)
            # sin(n*2Π/T*)
            self.cosx[i,j,:]  = torch.cos((i*2*np.pi/self.seasonal[j])*times)
            # cos(n*2Π/T*)
            
        self.sinx_infer=torch.zeros((season_num_,topk_freq,n_test))
        self.cosx_infer=torch.zeros((season_num_,topk_freq,n_test))
        
        self.F_ada=torch.nn.Linear(hid_size,hid_size//10)

    def _forward(self, roads, times,if_infer=False):
        """
        roads (batch_size,): road link ids.
        times (batch_size,): time steps.
        """
        ms = self.getu(roads)
        # ms:(batch_size, dim_size+bias_dim)

        mt = self.getv(times,if_infer)
        # mt: (T,dim_size+bias_dim)

        return torch.mm(ms, mt.t())
        # (batch_size, T)

    def forward(self, roads):
        """
        roads (batch_size, ): road link ids.
        """
        t = self.dims[1]
        # total time
        times = torch.arange(t, dtype=roads.dtype, device=roads.device)
        # times:[t]
        return self._forward(roads, times)
        # ret: (batch_size, T)

    def infer(self, n_step):
        """
        Infer the values of last n-step.
        """
        n, t = self.dims
        device_ = self.get_device

        roads = torch.arange(n).to(device_)
        # [N]
        times = torch.arange(t-n_step, t).to(device)
        # [n_step]
        for i in range(self.season_num):
          for j in range(self.topk_freq):
            # vector sin(ωt) and cos(ωt) for seasonality feature matrix
            self.sinx_infer[i,j,:] = torch.sin((i*2*np.pi/self.seasonal[j])*times)
            # sin(n*2Π/T*)
            self.cosx_infer[i,j,:]  = torch.cos((i*2*np.pi/self.seasonal[j])*times)
            # cos(n*2Π/T*)
        
        return self._forward(roads, times,True)
    
    def bias_loss(self, times):
        """
        times (batch_size,): time steps.
        maxlag: int
        """

        mt_bias = self.bias_t(times)
        # mt_bias:(T,bias_dim)
        
        max_len = int(max(self.lag_list))

        t = mt_bias.shape[0]
        if isinstance(t, int):
            pass
        else:
            t = int(t)
            
        tensor_len = t-max_len
        
        lst_lag = []
        
        for i in self.lag_list:
            lst_lag.append(mt_bias[max_len-i:max_len-i+tensor_len].clone())
            # for every time lag，we clone one corresponding part for autoregression
            # for example, if temporal length is 50  and time_lag is [1,3,5]
            # then the sequences we clone are (4.49),(2,47),(0,45)，which are all elements effecting (5,50)
        
        ret_bias_origin = mt_bias[max_len:max_len+tensor_len].clone()

        ret_var = self.modlst[0](lst_lag[0])
        for i in range(1, self.len_lag):
            ret_var = ret_var+self.modlst[i](lst_lag[i])
        
        return ret_var-ret_bias_origin
        # makes bias feature matrix follow a certain autoregression relationship

    @property
    def get_device(self):
        return self.s_embeddings.weight.device

    def getu(self, roads):
        device_ = self.get_device
        ms = self.s_embeddings(roads)
        # ms:(batch_size, dim_size)
        s_batch_size = ms.shape[0]
        # batch_size
        ones_s = torch.ones((s_batch_size, self.bias_dim)).to(device_)
        # ones_s: (batch_size,bias_dim)
        bias_s=self.s_embeddings_bias(roads)
        # bias_s:(batch_size,u_bias_dim)
        ms = torch.cat((ms, ones_s,bias_s), 1).to(device_)
        # ms:(batch_size, dim_size+bias_dim)

        return ms

    def getv(self, times,if_infer=False):
        device_ = self.get_device
        mt_trend = self.getv_trend(times)
        # mt_trend:(T,dim_size+bias_dim)
        mt_season = self.getv_season(times,if_infer)
        # mt_season :(T,dim_size+bias_dim)
        
        softmax_res=torch.softmax(self.param,dim=-1)
        
        mt = softmax_res[0]*mt_trend + softmax_res[1]*mt_season
        # mt: (T, dim_size)
        
        mt_bias = self.bias_t(times)
        # mt_bias:(T,bias_dim)
        ones_v = torch.ones((mt_season.shape[0], self.u_bias_dim)).to(device_)
        mt= torch.cat((mt,mt_bias,ones_v),1).to(device_)  
        # mt: (T,dim_size+bias_dim)

        return mt

    def getv_trend(self, times):
        # get v's trend matrix
        device_ = self.get_device
        mt_trend = self.t_embeddings_trend(times)
        # mt_trend:(T, dim_size)
        return mt_trend

    def getv_season(self,times,if_infer=False):
        # get v'season matrix
        device_ = self.get_device
        ret = torch.zeros(times.shape[0], self.hidden_size).to(device_)
        # [T,dim_size]

        if(if_infer==False):
         for i in range(self.season_num):
          for j in range(self.topk_freq):
            sin_t, cos_t = self.sinx[i][j],self.cosx[i][j]
            # [T], [T]

            sin_t = sin_t.detach()
            cos_t = cos_t.detach()
            
            sin_t=sin_t.to(device_)
            cos_t=cos_t.to(device_)

            emb_a=self.t_embeddings_season(torch.LongTensor([i*2+j*self.season_num]).to(device_))
            emb_b=self.t_embeddings_season(torch.LongTensor([i*2+1+j*self.season_num]).to(device_))
           
            ret = ret+torch.einsum('i,zj->ji',
                                   sin_t,
                                   emb_a).t()
            # ret[T,dim_size]

            ret = ret+torch.einsum('i,zj->ji',
                                   cos_t,
                                   emb_b).t()
        else:
         for i in range(self.season_num):
          for j in range(self.topk_freq):
            sin_t, cos_t = self.sinx_infer[i][j],self.cosx_infer[i][j]
            # [T], [T]

            sin_t = sin_t.detach()
            cos_t = cos_t.detach()
            
            sin_t=sin_t.to(device_)
            cos_t=cos_t.to(device_)

            emb_a=self.t_embeddings_season(torch.LongTensor([i*2+j*self.season_num]).to(device_))
            emb_b=self.t_embeddings_season(torch.LongTensor([i*2+1+j*self.season_num]).to(device_))
           
            ret = ret+torch.einsum('i,zj->ji',
                                   sin_t,
                                   emb_a).t()
            # ret[T,dim_size]

            ret = ret+torch.einsum('i,zj->ji',
                                   cos_t,
                                   emb_b).t()
        mt_season = ret
        # mt_season:(T, dim_size)
        return mt_season




def obsloss(model, x, roads):
    """
    X (N, T)
    roads (batch_size,)
    """
    x_ = x[roads]
    # (batch_size, T)
    device_ = model.get_device

    x_, roads = x_.to(device_), roads.to(device_)

    xhat = model(roads)
    # (batch_size, T)
    return obsLossF(xhat, x_)



def spatialloss(model: nn.Module, roads):
                
    """
    Laplacian constraint loss Tr(U^T L U).

    Args
        model: MF instance.
        L (N, N): Graph laplacian matrix.
    """
    n, device_ = model.dims[0], model.get_device
    roads=roads.to(device_)
    u = model.s_embeddings(roads)
    #print(u.shape)
    #u_s=model.F_ada(u)
    bias_u=model.s_embeddings_bias(roads)
    
    
    
    sg=torch.nn.Sigmoid()
    r=torch.nn.ReLU()
    sf=torch.nn.Softmax(dim=1)
    
    
    A=(bias_u@bias_u.T)
    A=sf(A)
    d=torch.sqrt(torch.sum(A,axis=1))
    A=(A/d.reshape(1,roads.shape[0]))/(d.reshape(roads.shape[0],1))
    #A=sg(A-torch.mean(A,axis=1))
    L=torch.diag(torch.sum(A,dim=1))-A

    return torch.einsum('dn,nd->', u.T, L@ u) / n

def l2loss(model: nn.Module,
           eta_1: float,
           eta_2: float):
    n, t = model.dims
    device_ = model.get_device
    u = model.getu(torch.arange(n).to(device_))
    v = model.getv(torch.arange(t).to(device_))
    return eta_1*torch.linalg.norm(u) + eta_2*torch.linalg.norm(v)


def arloss_bias(model, lambda_ar_):
    device_ = model.get_device
    n, t = model.dims
    times = torch.arange(t).to(device_)
    y = model.bias_loss(times)
    loss = lambda_ar_*torch.linalg.norm(y)
    return loss


def trend_loss(model, lambda_trend_):
    device_ = model.get_device
    n, t = model.dims
    times = torch.arange(t).to(device_)
    trend = model.t_embeddings_trend(times)
    trend=trend.T
    loss = trend[:, 1:]-trend[:, :-1]
    loss = torch.linalg.norm(loss)
    
    return lambda_trend_*loss


def train(
    optimizer: torch.optim.Optimizer, 
    num_epochs_: int,
    batch_size_: int,
    model: nn.Module, 
    x: Tensor,
    x_val_:Tensor,
    eta_: float,
    verbose_: bool,
    lambda_ar_,
    n_test_,
    lambda_trend_,
    lambda_spatial,
    x_unobs_
):
    def train_step(i, roads,x,flag_train):
        

        if eta_ > 0:
            l2_loss = l2loss(model, eta_, eta_)
        else:
            l2_loss = 0.0
        # L2 regression

        if lambda_ar_ > 0:
            loss_bias = arloss_bias(model, lambda_ar_)
        else:
            loss_bias = 0.0
        # loss func for bias feature matrix
        
        if lambda_trend_ > 0:
            loss_trend = trend_loss(model, lambda_trend_)
        else:
            loss_trend = 0.0
        #loss func for trend feature matrix
        
        if lambda_spatial > 0:
            loss_spatial = lambda_spatial*spatialloss(model, roads)
        else:
            loss_spatial = 0.0
        #loss func for spatial feature matrix
        

        loss = obsloss(model, x, roads)+ l2_loss + loss_bias + loss_trend+loss_spatial
        #print(obsloss(model, x, roads)  ,l2_loss , loss_bias , loss_trend ,loss,loss_spatial)
        #print('*'*30)
        if(flag_train):
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

        return loss.item()
    
    def train_epoch(idx_,x,flag_train):
        model.train()
        np.random.shuffle(idx_)
        # suffle channels
        n, running_losses = len(idx_), []
        for st_idx in range(0, n, batch_size_):
            # randomly choose one batch for training
            end_idx = min(st_idx + batch_size_, n)
            roads = torch.LongTensor(idx_[st_idx:end_idx])
            # [batch_size_]
            loss_this=train_step(st_idx//batch_size_, roads,x,flag_train)
            running_losses.append(loss_this)
            
        return np.nanmean(running_losses)

    idx = np.arange(x.shape[0])

    min_val_loss=np.inf
    
    for epoch in range(num_epochs_):
        running_loss = train_epoch(idx,x,True)
        
        
        val_loss=train_epoch(idx,x_val_,False)
        if(val_loss <min_val_loss):   
          min_val_loss=val_loss
          min_epo=epoch
          torch.save(model.state_dict(),args.save_path)
    print('min epo: ',min_epo)

    
@torch.no_grad()
def test(model, x_unobs, n_test):
    
    model.eval()
    print(n_test)
    xhat = model.infer(n_test)

    n, device_ = model.dims[0], model.get_device
    
    
    #u = model.s_embeddings(torch.arange(n).to(device_))
    ##bias_u=model.s_embeddings_bias(torch.arange(n).to(device_))
    #sg=torch.nn.Sigmoid()
    #r=torch.nn.ReLU()
    #sf=torch.nn.Softmax(dim=1)
    
    
    
    #A=r(bias_u@bias_u.T)
    #A=sf(A)
    #d=torch.sqrt(torch.sum(A,axis=1))
    #A=(A/d.reshape(1,n))/(d.reshape(n,1))
    
    
    
    #np.save('U_AUTIDER_center_west.npy',u.detach().cpu().numpy())
    #np.save('U_BIAS_AUTIDER_center_west.npy',bias_u.detach().cpu().numpy())
    #np.save('A_TimesNetTIDER_center_west.npy',A.detach().cpu().numpy())
    x = x_unobs[:, -n_test:].to(xhat.device)
    mask = torch.logical_not(x.isnan())
    
    y_test_, y_pred_ = x[mask], xhat[mask]
    
    
    return y_test_.cpu().numpy(), y_pred_.cpu().numpy()

def one_snapshot(
    x_obs_: Tensor,
    x_val: Tensor,
    x_unobs_: Tensor,
    n_test_,
    num_epochs_: int,
    batch_size_: int,
    dim_size_: int,
    lag_list_,
    lambda_ar_,
    bias_dim_,
    season_num_,
    seasonality_,
    lr_,
    n_test,
    lambda_trend_,
    lambda_spatial,
    topk_freq,
    u_bias_dim,
    device_: torch.device,
    eta_: Optional[float] = 0.0,
    verbose: Optional[bool] = False
):        


    model = TIDER(x_obs_.shape[0],
               x_obs_.shape[1],
               dim_size_,
               lag_list_,
               bias_dim_,
               season_num_,
               seasonality_,
               topk_freq,
               u_bias_dim,
               n_test).to(device_)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_)

    train(optimizer,
          num_epochs_,
          batch_size_,
          model,
          x_obs_,
          x_val,
          eta_,
          verbose,
          lambda_ar_,
          n_test_,
          lambda_trend_,
          lambda_spatial,
          x_unobs_
    )
          
    model = TIDER(x_obs_.shape[0],
               x_obs_.shape[1],
               dim_size_,
               lag_list_,
               bias_dim_,
               season_num_,
               seasonality_,
               topk_freq,
               u_bias_dim,
               n_test).to(device_)
    model.load_state_dict(torch.load(args.save_path))
    #print('ok')
    softmax_re=torch.softmax(model.param,dim=-1)
    #print('ok')
    weight_0=softmax_re[0].item()
    weight_1=softmax_re[1].item()
    print(weight_0,weight_1)
    
    return test(model, x_unobs_, n_test_)


def update_obs(x_obs_, x_unobs_):
    """
    Merge the observation in `X_unobs` into `X_obs` except the latest time step.
    """
    observed = torch.logical_not(torch.isnan(x_unobs_))
    observed[:, -1] = False
    x_obs_[observed] = x_unobs_[observed]
    return x_obs_

def obsLossF(Xhat: Tensor, X: Tensor):
    """
    Xhat (n, m): Inferred matrix.
    X (n, m): Partially observed matrix with nan indicating missing values.
    """
    mask = torch.logical_not(X.isnan())
    return f.mse_loss(Xhat[mask], X[mask])

start = time.time()

np.random.seed(123)
torch.manual_seed(123)
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--save_path',default="ADATIDER.pt")
#parser.add_argument('--datadir',default='/home/shuailiu/TIDER/dataset/London_6mon.npy')
parser.add_argument('--datadir',default='../dataset/guangzhou_new.txt')
#parser.add_argument('--datadir',default='../dataset/west.txt')
#parser.add_argument('--datadir',default='/home/shuailiu/TIDER/dataset/solar.txt')
parser.add_argument('--device', default='cuda:1')
parser.add_argument('--valid', default=0.1,type=float)
parser.add_argument('--drop_rate', default=0.2,type=float)
parser.add_argument('--eta', default=1e-2,type=float)
#parser.add_argument('--n_test', default=52560, type=int)
parser.add_argument('--n_test', default=72, type=int)
parser.add_argument('--num_epochs', default=5000, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--dim_size', default=50, type=int)
parser.add_argument('--lag_list', default='list(range(6))')
parser.add_argument('--lambda_ar', default=0.1, type=float)
parser.add_argument('--bias_dimension', default=4, type=int)
parser.add_argument('--season_num', default=10, type=int)
#parser.add_argument('--seasonality', default=168, type=float)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--lambda_trend', default=0.1, type=float)
parser.add_argument('--lambda_spatial', default=0.01, type=float)
parser.add_argument('--topk_freq',default=10,type=int)
parser.add_argument('--u_bias_dim',default=4,type=int)

args = parser.parse_args()

if __name__ == "__main__":
    start = time.time()
    


    data = args.datadir
    if(os.path.splitext(data)[1]=='.txt'):
      data_=open(data,'r')

      x=np.loadtxt(data_,delimiter=',')
    elif(os.path.splitext(data)[1]=='.npy'):
      x=np.load(data)[:100,:1000]
    print(x.shape)
    #load time-series data
    
    trn=np.copy(x)
    val=np.copy(x)
    tst=np.copy(x)
     
    c_len=list(range(x.shape[1]))
    
    for i in range(x.shape[0]):
            random.shuffle(c_len)
            trn[i][c_len[:int(x.shape[1]*args.valid)]]=np.nan
            trn[i][c_len[int(x.shape[1]*(1-args.drop_rate)):-1]]=np.nan
            
            val[i][c_len[int(x.shape[1]*args.valid):-1]]=np.nan
            
            tst[i][c_len[:int(x.shape[1]*(1-args.drop_rate))]]=np.nan
  
       
         
    X_obs, X_val, X_unobs = torch.FloatTensor(trn), torch.FloatTensor(val),torch.FloatTensor(tst)  
         
    #split data into training, validation and testing

    
    #device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    device=torch.device(args.device)
    eta = args.eta
    lrate = args.learning_rate
    bias_dim = args.bias_dimension
    lambda_trend = args.lambda_trend
    lambda_spatial = args.lambda_spatial
    season_num = args.season_num
    #seasonality = args.seasonality
    n_test = args.n_test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    dim_size = args.dim_size
    lag_list = eval(args.lag_list)
    lambda_ar = args.lambda_ar
    u_bias_dim=args.u_bias_dim
    #load the hyperparameters
    
    
    #get topk frequencies 【those with largest amplitude】
    topk_freq=args.topk_freq
    seasonality=get_top_freq(x,topk_freq)
    topk_freq=len(seasonality)
    print(seasonality)

    y_test, y_pred = one_snapshot(
            X_obs, 
            X_val,
            X_unobs,
            n_test,
            num_epochs, 
            batch_size, 
            dim_size, 
            lag_list,
            lambda_ar,
            bias_dim,
            season_num,
            seasonality,
            lrate,
            n_test,
            lambda_trend,
            lambda_spatial,
            topk_freq,
            u_bias_dim,
            device,
            eta
        )
    print(evaluated_message(y_test, y_pred))
                           
    end = time.time()
    print('Running time: %d seconds' % (end - start))
    print(args)
    print('\n'*10)
