
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import h5py
import os
import sys
import warnings
warnings.filterwarnings("ignore")

def initialize_parameters(Lin, Lhid):
    
    np.random.seed(443)
    Lout=Lin
    w_0 =np.sqrt(6/(Lin+Lhid))
    w_1=np.sqrt(6/(Lhid+Lout))
    
    W1 = np.random.uniform(-w_0,w_0,(Lin,Lhid))
    b1 = np.random.uniform(-w_0,w_0,(1,Lhid))
    
    
    W2 = W1.T # at the same time W2=W1.T 
    b2 = np.random.uniform(-w_1,w_1,(1,Lout))
    return (W1, W2, b1, b2), (0,0,0,0)

sigmoid = lambda x:np.exp(x)/ (1+np.exp(x))
derSigmoid= lambda x: sigmoid(x)*(1-sigmoid(x))

def forward(W_e, data):
    W1, W2, b1, b2 = W_e
    
    W1_ = data.dot(W1)+b1
    z = sigmoid(W1_)
    dz= derSigmoid(W1_)
    
    
    W2_ = z.dot(W2)+b2
    z2 = sigmoid(W2_)
    dz2= derSigmoid(W2_)
    
    cac=(data,dz,dz2)

    return z, z2,cac

def aeCost(W_e, data, params):
    (Lin, Lhid, lmb, beta, rho) = params
    W1, W2, b1, b2 = W_e
    N= len(data)
    
    hid, out, (_,d_hid,d_out) = forward(W_e, data)
    hid_mean = hid.mean(axis=0,keepdims=True)

    ASE = (1/(2*N)) * np.sum(np.power((data - out),2)) #mse

    TYK = (lmb/2) * (np.sum(W1**2) + np.sum(W2**2)) # Tykhonov
    KL = rho*np.log(rho/hid_mean) + (1-rho)*np.log((1-rho)/(1-hid_mean))
    KL =  beta * KL.sum() # Kullback-Leibler
    
    J = ASE + TYK + KL ## J_ae
    
    d_ASE=-(data-out)/N
    d_TYK=W1*lmb , W2*lmb
    d_KL=beta*(- rho/hid_mean + (1-rho)/(1 - hid_mean))/N
    
    d1 = d_ASE * d_out
    dW2 = hid.T.dot(d1) + d_TYK[1]
    db2 = d1.sum(axis=0, keepdims=True)

    d2 = d_hid * (d1 .dot( W2.T) + d_KL)

    dW1 = data.T .dot(d2) + d_TYK[0]
    db1 = d2.sum(axis=0, keepdims=True)    
    
    
    J_grad=(dW1,dW2,db1,db2)
    return J, J_grad

def update_parameters(We, mW, dW, m, l_rate):
    mW = m * np.array(dW, dtype=object) + l_rate * np.array(mW, dtype=object)
        
    We = np.array(We, dtype=object) - mW
        
    We = tuple(We)
    mW = tuple(mW)
    
    return We, mW

def solver( data, params, eta, alpha, epoch, batch): 

    loss_list = []
    if batch is None:
        batch = len(data)

    Lin = params[0]
    Lhid = params[1]
    
    We, mWe = initialize_parameters(Lin,  Lhid)

    iter_ep = int(len(data) / batch)
    # j_prev=0  ##early stopping but not used
    for k in range(epoch):
        J_sum = 0

        temp_start=0
        temp_end = batch


        
        for j in range(iter_ep):

            data_temp = data[temp_start:temp_end]

            J, Jgr = aeCost(We, data_temp, params)
            
            
            
            We, mWe = update_parameters(We, mWe, Jgr, eta, alpha)
            
            temp_start = temp_end
            temp_end += batch
            J_sum += J
        
        J_sum = J_sum/iter_ep
        #earl=abs(J_sum-j_prev)/abs(J_sum) ## early stopping but not used
        #j_prev=J_sum ##  early stopping but not used
        
        #if earl<0.0001:
        #    break

        print("Loss: {:.2f} [Epoch {} of {}]".format(J_sum, k+1, epoch))
        loss_list.append(J_sum)

    return We, loss_list

def plot_weights(we): 
    (w1,w2,b1,b2) = we
    fig=plt.figure(figsize=(18, 16))
    plot_shape = int(np.sqrt(w1.shape[1]))
    for i in range(w1.shape[1]):
        plt.subplot(plot_shape,plot_shape,i+1)
        plt.imshow(np.reshape(w1[:,i],(16,16)), cmap='gray')
        plt.axis('off')
    plt.show()
    
def label_1h(y, size):
    output = np.zeros((len(y), size))
    for q in range(len(y)):
        temp = np.zeros(size)
        temp[y[q]-1] = 1               
        output[q,:] = temp
    return output

def data_1h(x, size):
    output = np.zeros((len(x), x.shape[1], size))
    for q in range(len(x)):
        for p in range(x.shape[1]):
            temp = np.zeros(size)
            temp[x[q,p]-1] = 1             
            output[q,p,:] = temp
    return output

class Layer: 
    def __init__(self,dim_in,neur_N,avg,std, act):
        self.dim_in = dim_in
        self.neur_N = neur_N
        self.act = act
        self.prevU = 0
        self.Last_Act=None
        self.err_layer=None
        self.delta_layer=None
        if self.act == 'softmax' or self.act == 'sigmoid':
            self.all_W= np.random.normal(avg,std, (neur_N,dim_in+1))
            self.weight=self.all_W[:,:-1]
            self.bias=self.all_W[:,-1:]
        else:
            self.D = dim_in
            self.S_dict = neur_N
            self.weight = np.random.normal(avg, std, (self.S_dict,self.D))        
        
    def act_f(self, x):

        if(self.act == 'softmax'):
            e_x = np.exp(x - np.max(x))
            return e_x/np.sum(e_x, axis=0)
        
        elif(self.act == 'sigmoid'):
            return np.exp(2*x)/(1+np.exp(2*x))        
        else:
            return x
        
    def act_N(self,x):     
        if self.act == 'sigmoid' or self.act == 'softmax':                
            samp_N = x.shape[1]
            inp_temp = np.r_[x, [np.ones(samp_N)*-1]]    
            self.Last_Act = self.act_f(self.all_W.dot(inp_temp))
            
        else:
            Embed = np.zeros((x.shape[0],x.shape[1], self.D))
            for m in range(Embed.shape[0]):
                Embed[m,:,:] = self.act_f(x[m,:,:].dot(self.weight))
            Embed = Embed.reshape((Embed.shape[0], Embed.shape[1] * Embed.shape[2]))
            self.Last_Act = Embed.T 
        return self.Last_Act        

    def act_der(self, x):
        if(self.act == 'sigmoid' or self.act == 'softmax'):
            return x*(1-x)
        else:
            return np.ones(x.shape)




class Neural: 
    def __init__(self):
        self.lyrs=[]
     
    def layer_add(self,layer):
        self.lyrs.append(layer)
 
    def Forward(self,data_train):
        IN=data_train    
        for layer in self.lyrs:
            IN=layer.act_N(IN)
        return IN

    def Back(self,l_rate,size_batch,data_train,label_train,momentum):   
        out_forwa = self.Forward(data_train)
        for i in reversed(range(len(self.lyrs))):
            lyr = self.lyrs[i]
      
            if(lyr == self.lyrs[-1]):
                lyr.delta_layer=label_train.T-out_forwa     
            else:
                layer_next = self.lyrs[i+1]
                lyr.err_layer = np.matmul(layer_next.weight.T, layer_next.delta_layer)
                der=lyr.act_der(lyr.Last_Act)               
                lyr.delta_layer=der*lyr.err_layer
             
        for i in range(len(self.lyrs)):
            lyr = self.lyrs[i]
            if(i == 0):
                inp_temp = data_train
            else:
                samp_N = self.lyrs[i - 1].Last_Act.shape[1]
                inp_temp = np.r_[self.lyrs[i - 1].Last_Act, [np.ones(samp_N)*-1]]
                
                
            if(lyr.act == 'sigmoid' or lyr.act == 'softmax'):
                update =  l_rate*np.matmul(lyr.delta_layer, inp_temp.T)
                lyr.all_W+= update/size_batch + (momentum*lyr.prevU)
            else:          
                emb_delta = lyr.delta_layer.reshape((3,size_batch,lyr.D))
                inp_temp = np.transpose(inp_temp, (1,0,2)) 
                update = np.zeros((inp_temp.shape[2], emb_delta.shape[2]))
                for i in range(emb_delta.shape[0]):
                    update += l_rate * np.matmul(inp_temp[i,:,:].T, emb_delta[i,:,:])
                update = update
                lyr.weight += update/size_batch + (momentum*lyr.prevU)
            lyr.prevU = update/size_batch
         
    def Train(self,l_rate,size_batch,data_train,label_train, data_test, lbl_test, num_ep,momentum,S_dict):
        losses = []    
        temp_loss=0
        for ep in range(num_ep):
            
            print("Ep:",ep)
            indexing=np.random.permutation(len(data_train))
            data_train=data_train[indexing,:]
            label_train=label_train[indexing]
            batch_N = int(np.floor(len(data_train)/size_batch)) 
            for j in range(batch_N):
                data_one_hot = data_1h(data_train[j*size_batch:size_batch*(j+1),:], S_dict)
                label_one_hot = label_1h(label_train[j*size_batch:size_batch*(j+1)], S_dict)
                self.Back(l_rate,size_batch,data_one_hot,label_one_hot,momentum)         
         
            out_val = self.Forward(data_test)
            los_cros = - np.sum(np.log(out_val) * lbl_test.T)/out_val.shape[1]
            print('C-E Error ', los_cros)
            losses.append(los_cros)
            print(abs(los_cros-temp_loss)/abs(los_cros))
            if abs(los_cros-temp_loss)/abs(los_cros)<0.005:
                print("stopped based on cross-entropy")
                break
            temp_loss=los_cros

        return losses
             
 
    def predict(self, inputIMG, k):
        out = self.Forward(inputIMG)
        return np.argsort(out, axis=0)[:,0:k]

    
class lyr: #lyr Class for RNN
    def __init__(self,dim_in,neur_N,act,beta):
        self.dim_in = dim_in
        self.neur_N = neur_N
        self.act = act
        self.beta=beta 
        self.prevU = 0
        self.prevU_RNN = 0    
        self.Last_Act=None
        self.err_lyr=None
        self.delta_lyr=None
        
        self.XUD=np.sqrt(6/(dim_in+neur_N))
        self.all_W =np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N))
        self.W2=np.random.uniform(-self.XUD,self.XUD,(neur_N,neur_N))

        

        
    def act_f(self, x):
          
        if(self.act == 'softmax'):
            e_x = np.exp(x - np.max(x))
            return e_x/np.sum(e_x, axis=1, keepdims=True)
        elif(self.act == 'hyperbolic'):
            return np.tanh(x*self.beta)
        elif(self.act=="sigmoid"):
            return np.exp(2*x)/(1+np.exp(2*x)) 
        elif(self.act=="relu"):
            return np.maximum(0,x)
        else:
            return x
        
    def act_N(self,x):  
        samp_N = len(x)
        inp_temp = np.concatenate((x, -np.ones((samp_N, 1))), axis=1)
        self.Last_Act = self.act_f(inp_temp.dot(self.all_W))
        return self.Last_Act   
    
    def rec_Act(self,x,hid):
        samp_N = len(x)
        inp_temp = np.concatenate((x, -np.ones((samp_N, 1))), axis=1)
        last= hid.dot(self.W2)+ inp_temp.dot(self.all_W)
        self.Last_Act = self.act_f(last)
        return self.Last_Act   
        
    def act_derv(self, x):
        if(self.act=="sigmoid" or self.act == 'softmax'):
            return x*(1-x)
        elif (self.act=="relu"):
            return (x>0)*1 
        elif(self.act == 'hyperbolic'):
            return self.beta*(1-x*x) 
        else:
            return np.ones(x.shape)

class RNN_:
    def __init__(self,data_train):
        self.samp_T= data_train.shape[1]
        self.del_rec=np.empty((32, self.samp_T, 128))
        self.loss_rec=np.empty((32, self.samp_T, 128))
        self.p_hid=np.zeros((32, 128))
        self.lyrs=[]
        
    def lyr_add(self,lyr):
        self.lyrs.append(lyr)
    
    def Forward(self,data_train):#Foward Prop
        samp_N, samp_T, D = data_train.shape
        inp=np.empty((samp_N, samp_T, 128))  
        self.p_hid=np.zeros((samp_N, 128))
        for t in range(samp_T): #all time units are needed
            x = data_train[:, t]
            inp[:, t]=self.lyrs[0].rec_Act(x,self.p_hid) 
            self.p_hid=inp[:, t]
        output = inp[:, -1] 
            
        for lyr in self.lyrs[1:len(self.lyrs)]: # MLP 
            output=lyr.act_N(output) #last  sample  contain memory
        return inp,output

    def back(self,l_rate,size_batch,data_train,lab_train,momentum):   
        inp,output = self.Forward(data_train)     
        out_forw = output
        for i in reversed(range(len(self.lyrs))):# backprop til recurrent
            lyr = self.lyrs[i]
            #out layer
            if(lyr == self.lyrs[-1]):
                lyr.delta_lyr=lab_train-out_forw    
            else :
                layer_next = self.lyrs[i+1]
                lyr.err_lyr = layer_next.delta_lyr.dot(layer_next.all_W[:len(layer_next.all_W)-1].T)
                derv=lyr.act_derv(lyr.Last_Act)               
                lyr.delta_lyr=derv*lyr.err_lyr   
                if (lyr == self.lyrs[0]):
                    self.loss_rec[:,-1]=lyr.err_lyr  
                    self.del_rec[:,-1]=lyr.delta_lyr

        d_all_weight=0
        d_hid_weight=0
        samp_N, samp_T, D = data_train.shape
  
        for t in reversed(range(samp_T)):
            lyr=self.lyrs[0]
            if t > 0:
                u = inp[:, t-1]
            else:
                u = np.zeros((samp_N, 128))      
                
            derv=lyr.act_derv(u)
            d_hid_weight+=u.T.dot(self.del_rec[:,t])            
            d_all_weight+=np.concatenate((data_train[:,t-1], -np.ones((len(data_train[:,t-1]), 1))), axis=1).T.dot(self.del_rec[:,t])
            
            # Recc dlt updte
            self.loss_rec[:,t-1]=self.del_rec[:,t].dot(lyr.W2.T)
            self.del_rec[:,t-1]=self.loss_rec[:,t-1]*derv

        
        for i in range(len(self.lyrs)): #update all weights
            lyr = self.lyrs[i]
            if(i == 0):                
                lyr.prevU =  d_all_weight*l_rate/(150*size_batch)
                lyr.prevU_RNN =  d_hid_weight*l_rate/(150*size_batch)
                lyr.all_W+=  (momentum*lyr.prevU) +lyr.prevU 
                lyr.W2+= (momentum*lyr.prevU_RNN) +lyr.prevU_RNN
                
            else:      
                samp_N = len(self.lyrs[i - 1].Last_Act)
                inp_temp=np.concatenate((self.lyrs[i - 1].Last_Act, -1*np.ones((samp_N, 1))), axis=1)   
                lyr.prevU =  inp_temp.T.dot(lyr.delta_lyr)*l_rate/size_batch
                lyr.all_W+= lyr.prevU*momentum +lyr.prevU
            
    def train(self,l_rate,size_batch,data_train,lab_train, inp_test, lab_test, N_ep,momentum):
        cr_los_list = []   
        train_res=[]
        for ep in range(N_ep):
            print("ep:",ep)
            ind=np.random.permutation(len(data_train)) 
            data_train=data_train[ind]
            lab_train=lab_train[ind]
            batch_N = int(len(data_train)/size_batch) 
            for j in range(batch_N):
                train_data = data_train[j*size_batch:(j+1)*size_batch]
                train_labels = lab_train[j*size_batch:(j+1)*size_batch]
                self.back(l_rate,size_batch,train_data,train_labels,momentum)         
            _, out_val = self.Forward(inp_test)
            _, out_train = self.Forward(data_train)
            cr_los =  np.sum(-lab_test*np.log(out_val))/len(out_val)
            cr_los1 =  np.sum(-lab_train*np.log(out_train))/len(out_train)
            print('C-E Error of Validation', cr_los) 
            print('C-E Error Error of Train', cr_los1)   
            cr_los_list.append(cr_los)
            train_res.append(cr_los1)
        return cr_los_list, train_res
    
    def Predict(self,inps,realout):
        _,out = self.Forward(inps)
        out = out.argmax(axis=1)
        realout = realout.argmax(axis=1)
        return ((out == realout).mean()*100)
    
    def conf(self,inp,outp):
        _,pred= self.Forward(inp)
        pred = pred.argmax(axis=1)
        outp = outp.argmax(axis=1)
        q = len(np.unique(outp))
        conf=np.zeros((q,q))
        for p in range(len(outp)):
            conf[outp[p]][pred[p]] += 1
        return conf
    
class LSTM_lyr: #LSTM lyr Class
    def __init__(self,dim_in,neur_N,beta):
        self.dim_in = dim_in
        self.neur_N = neur_N
        self.beta=beta
        self.Last_Act=None
        self.err_lyr=None
        self.delta_lyr=None
        self.prevU_f, self.prevU_i, self.prevU_c, self.prevU_o= 0,0,0,0
        
        self.XUD=np.sqrt(6/(dim_in+neur_N))
        

        self.Wf = np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N)) # forget gate
        self.Wi = np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N)) # input gate
        self.Wc = np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N))  # cell gate
        self.Wo = np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N))  # output gate
        

    def act_f(self, x,act):       
        if(act == 'softmax'):
            e_x = np.exp(x - np.max(x))
            return e_x/np.sum(e_x, axis=1, keepdims=True)
        elif(act == 'tanh'):
            return np.tanh(x*self.beta)
        elif(act=="sigmoid"):
            return np.exp(2*x)/(1+np.exp(2*x)) 
        elif(act=="relu"):
            return np.maximum(0,x)
        else:
            return x
        

    def act_N(self,x, w, act):  
        samp_N = len(x)
        inp_temp = np.concatenate((x, -np.ones((samp_N, 1))), axis=1)
        self.Last_Act = self.act_f(inp_temp.dot(w),act)
        return self.Last_Act   
    
    def rec_Act(self,x,hid, act):
        samp_N = len(x)
        inp_temp = np.concatenate((x, -np.ones((samp_N, 1))), axis=1)
        last=hid.dot(self.W2) + inp_temp.dot(self.all_W)
        self.Last_Act = self.act_f(last,act)
        return self.Last_Act   
        
    def act_derv(self, x,act):
        if(act=="sigmoid" or act == 'softmax'):
            return x*(1-x)
        elif (act=="relu"):
            return (x>0)*1 
        elif(act == 'tanh'):
            return self.beta*(1-x*x) 
        else:
            return np.ones(x.shape)
        
class LSTM_:
    def __init__(self,data_train): 

        self.lyrs=[]
        
    def lyr_add(self,lyr): 
        self.lyrs.append(lyr)
    
    def Forward(self,data_train):

        sampN, sampT, sampD = data_train.shape
        sampH=128 
        lyr = self.lyrs[0]      
        
        mem = np.empty((sampN, sampT, sampH))
        i_t = mem
        f_t = mem
        c_t = mem
        o_t = mem
        tanhc = mem
        
        
        h_t_1=np.zeros((sampN, sampH))
        c_prv=h_t_1

        
        z = np.empty((sampN, sampT, sampD + sampH))
        
        
        
        #Apply  functions
        for i in range(sampT):
            z[:, i] = np.concatenate((h_t_1, data_train[:, i]),axis=1)
            zt = z[:, i]
            
            i_t[:, i] = lyr.act_N(zt, lyr.Wi, "sigmoid")
            f_t[:, i] = lyr.act_N(zt , lyr.Wf, "sigmoid")
            c_t[:, i] = lyr.act_N(zt, lyr.Wc, "tanh")
            o_t[:, i] = lyr.act_N(zt, lyr.Wo, "sigmoid")

            mem[:, i] = c_t[:, i]*i_t[:, i] +  c_prv*f_t[:, i] 
            c_prv=mem[:, i]
            
            tanhc[:, i] = lyr.act_f(c_prv, "tanh")
            h_t_1 = o_t[:, i] * tanhc[:, i]
            

            cac = {"z_summ": z, #Summation of h_t-1 and x_t
                     "memory": mem,  #Memory
                     "tanhc": (tanhc), # tanh memor
                     "f_t": f_t, # f_t out
                     "i_t": (i_t), # i_t out
                     "c_t": (c_t),# c_t out
                     "o_t": (o_t)}# o_t out
                        

        for lyr in self.lyrs[1:len(self.lyrs)]: 
            #For MLP lyrs
            h_t_1=lyr.act_N(h_t_1) 
        OUT= h_t_1
        return cac,OUT

    def back(self,l_rate,size_batch,data_train,lab_train,momentum):   
        cac,output = self.Forward(data_train)     
        out_forw = output
        z = cac["z_summ"]
        c=cac["memory"]
        tanhc=cac["tanhc"]
        f_t=cac["f_t"]
        i_t=cac["i_t"]
        c_t=cac["c_t"]
        o_t=cac["o_t"]
        
        for q in reversed(range(len(self.lyrs))):# backprop til LSTM part
            lyr = self.lyrs[q]
            
            if(lyr == self.lyrs[-1]): #output
                lyr.delta_lyr=lab_train-out_forw    
            elif(lyr==self.lyrs[0]):
                layer_next = self.lyrs[q+1]
                lyr.err_lyr = layer_next.delta_lyr.dot(layer_next.all_W[0:len(layer_next.all_W)-1].T)
                lyr.delta_lyr=lyr.err_lyr
            
            else:
                layer_next = self.lyrs[q+1]
                lyr.err_lyr = layer_next.delta_lyr.dot(layer_next.all_W[0:len(layer_next.all_W)-1].T)
                derv=lyr.act_derv(lyr.Last_Act)               
                lyr.delta_lyr=derv*lyr.err_lyr
        
        
        dWf, dWi, dWc, dWo = 0,0,0,0  #init grads to zero
        H=128
        T = z.shape[1]
        samp_N = len(data_train)
        
        init_lyr=self.lyrs[0]
        delta=init_lyr.delta_lyr
        
        for t in reversed(range(T)):#BPTT OF LSTM
            u = z[:, t]

            if t > 0:
                c_prv = c[:, t - 1]
            else:
                c_prv = 0
        
                
            dc = delta * o_t[:, t] * init_lyr.act_derv(tanhc[:, t],"tanh")
            
            dc_t = dc * i_t[:, t] * init_lyr.act_derv(c_t[:, t],"tanh")
            di_t = dc * c_t[:, t] * init_lyr.act_derv(i_t[:, t],"sigmoid")
            df_t = dc * c_prv * init_lyr.act_derv(f_t[:, t],"sigmoid")
            do_t = delta * tanhc[:, t] * init_lyr.act_derv(o_t[:, t],"sigmoid")


            dWc += np.concatenate((u, -np.ones((samp_N, 1))), axis=1).T.dot(dc_t)
            dWi += np.concatenate((u, -np.ones((samp_N, 1))), axis=1).T.dot(di_t)
            dWf += np.concatenate((u, -np.ones((samp_N, 1))), axis=1).T.dot(df_t)
            dWo += np.concatenate((u, -np.ones((samp_N, 1))), axis=1).T.dot(do_t)
            
            # upd gradients
            duc = dc_t.dot(init_lyr.Wc.T[:, :H])
            dui = di_t.dot(init_lyr.Wi.T[:, :H])
            duf = df_t.dot(init_lyr.Wf.T[:, :H])
            duo = do_t.dot(init_lyr.Wo.T[:, :H])            
            
            delta = duc+dui+duf+duo
            
        
        for i in range(len(self.lyrs)): #update the Weights
            lyr = self.lyrs[i]
            if(i == 0): 
                
                up_f,up_i,up_c,up_o=np.array([dWf,dWi,dWc,dWo])*l_rate/size_batch
  
                lyr.Wf+= up_f + lyr.prevU_f*momentum
                lyr.Wi+= up_i + lyr.prevU_i*momentum
                lyr.Wc+= up_c + lyr.prevU_c*momentum
                lyr.Wo+= up_o + lyr.prevU_o*momentum
                
                lyr.prevU_f ,lyr.prevU_i,lyr.prevU_c,lyr.prevU_o=np.array([up_f,up_i,up_c,up_o])
          
            else:      
                samp_N = len(self.lyrs[i - 1].Last_Act)
                inp_temp=np.concatenate((self.lyrs[i - 1].Last_Act, -np.ones((samp_N, 1))), axis=1)   
                upd =  (inp_temp.T.dot(lyr.delta_lyr))*l_rate/size_batch
                lyr.all_W+= upd + lyr.prevU*momentum
                lyr.prevU = upd      
                
    def train(self,l_rate,size_batch,data_train,lab_train, inp_test, lab_test, N_ep,momentum):
        cr_los_list = []   
        train_res=[]
        for ep in range(N_ep):
            print("ep:",ep)
            ind=np.random.permutation(len(data_train))
         
            data_train=data_train[ind]
            lab_train=lab_train[ind]
            batch_N = int(len(data_train)/size_batch) 
            for j in range(batch_N):
                train_data = data_train[j*size_batch:size_batch*(j+1)]
                train_labels = lab_train[j*size_batch:size_batch*(j+1)]
                self.back(l_rate,size_batch,train_data,train_labels,momentum)         
            _, out_val = self.Forward(inp_test)
            _, out_train = self.Forward(data_train)
            cr_los = np.sum(-np.log(out_val) * lab_test)/len(out_val)
            cr_los1 = np.sum(-np.log(out_train) * lab_train)/len(out_train)
            print('C-E Error of Validation', cr_los)
            print('C-E Error of Train', cr_los1)
            cr_los_list.append(cr_los)
            train_res.append(cr_los1)
        return cr_los_list, train_res
    
    def Predict(self,inps,realout):
        _,out = self.Forward(inps)
        out = out.argmax(axis=1)
        realout = realout.argmax(axis=1)
        return ((out == realout).mean()*100)
    
    
    def conf(self,inp,outp):
        _,pred= self.Forward(inp)
        pred = pred.argmax(axis=1)
        outp = outp.argmax(axis=1)
        q = len(np.unique(outp))
        conf=np.zeros((q,q))
        for p in range(len(outp)):
            conf[outp[p]][pred[p]] += 1
        return conf
    
class GRU_lyr: #GRU lyr Class
    def __init__(self,dim_in,neur_N,beta):
        self.dim_in = dim_in
        self.neur_N = neur_N
        self.beta = beta
        
        self.Last_Act=None
        self.err_lyr=None
        self.delta_lyr=None
        self.prevU_Wz,self.prevU_Wr,self.prevU_Wh=0,0,0
        self.prevU_Uz,self.prevU_Ur,self.prevU_Uh=0,0,0
        
        self.XUD=np.sqrt(6/(dim_in+neur_N))  
        self.w1=np.sqrt(6/(neur_N+neur_N)) 

        self.Uz = np.random.uniform(-self.w1, self.w1, size=(neur_N, neur_N))
        self.Wz = np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N))

        self.Ur = np.random.uniform(-self.w1, self.w1, size=(neur_N, neur_N))
        self.Wr = np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N))

        self.Uh = np.random.uniform(-self.w1, self.w1, size=(neur_N, neur_N))
        self.Wh = np.random.uniform(-self.XUD,self.XUD,(dim_in+1,neur_N))     
        
        
    def act_f(self, x,act):       
        if(act == 'softmax'):
            e_x = np.exp(x - np.max(x))
            return e_x/np.sum(e_x, axis=1, keepdims=True)
        elif(act == 'tanh'):
            return np.tanh(x*self.beta)
        elif(act=="sigmoid"):
            return np.exp(x)/(1+np.exp(x)) 
        elif(act=="relu"):
            return np.maximum(0,x)
        else:
            return x
        
    def act_N(self,x, w, h, u, act):  
        samp_N = len(x) 
        inp_temp = np.concatenate((x, -np.ones((samp_N, 1))), axis=1)
        self.Last_Act = self.act_f(inp_temp.dot(w)+h.dot(u),act)
        return self.Last_Act   
        
    def act_derv(self, x,act):
        if(act=="sigmoid" or act == 'softmax'):
            return x*(1-x)
        elif (act=="relu"):
            return (x>0)*1 
        elif(act == 'tanh'):
            return self.beta*(1-x*x) 
        else:
            return np.ones(x.shape)


class GRU_:
    def __init__(self,data_train): 
        

        self.lyrs=[]
    def lyr_add(self,lyr): 
        self.lyrs.append(lyr)
    
    def Forward(self,data_train): 

        lyr = self.lyrs[0]    #GRU First lyr 
        sampN, sampT, _ = data_train.shape
        sampH=128        
        h_t_1=np.zeros((sampN, sampH))
        h_t = np.empty((sampN, sampT, sampH))
        h_t_t = np.empty((sampN, sampT, sampH)) 
        
        z_t = np.empty((sampN, sampT, sampH))
        r_t = np.empty((sampN, sampT, sampH))
        
        #apply funcs
        for t in range(sampT):            
            x = data_train[:, t]
            z_t[:, t] = lyr.act_N(x,  lyr.Wz, h_t_1, lyr.Uz , "sigmoid")
            r_t[:, t] = lyr.act_N(x, lyr.Wr, h_t_1,  lyr.Ur , "sigmoid")
            h_t_t[:, t] = lyr.act_N(x, lyr.Wh, (r_t[:, t] * h_t_1), lyr.Uh, "tanh")
            h_t[:, t] = (1 - z_t[:, t]) * h_t_1 + z_t[:, t] * h_t_t[:, t]
            h_t_1 = h_t[:, t]

            cac = {"z_t": z_t, 
                     "r_t": r_t,  
                     "h_t_t": (h_t_t), 
                     "h_t": h_t}
                        

        for ly in self.lyrs[1:len(self.lyrs)]: # MLP 
            h_t_1=ly.act_N(h_t_1) 
            
        oup= h_t_1
        return cac,oup

    def back(self,l_rate,size_batch,data_train,lab_train,momentum):   
        cac,OUT = self.Forward(data_train)     
        out_forw = OUT
        z_t = cac["z_t"]
        r_t=cac["r_t"]
        h_t_t=cac["h_t_t"]
        h_t=cac["h_t"]

        
        for q in reversed(range(len(self.lyrs))):# backprop til GRU
            lyr = self.lyrs[q]
            #outputlyr
            if(lyr == self.lyrs[-1]):
                lyr.delta_lyr=lab_train-out_forw    
            elif(lyr==self.lyrs[0]):
                layer_next = self.lyrs[q+1]
                lyr.err_lyr = layer_next.delta_lyr.dot(layer_next.all_W[0:len(layer_next.all_W)-1].T)
                lyr.delta_lyr=lyr.err_lyr
            
            else:
                layer_next = self.lyrs[q+1]
                lyr.err_lyr = layer_next.delta_lyr.dot(layer_next.all_W[0:len(layer_next.all_W)-1].T)
                derv=lyr.act_derv(lyr.Last_Act)               
                lyr.delta_lyr=derv*lyr.err_lyr
        # initialize gradients to zero
        dWz,dWr,dWh = 0,0,0
        dUz,dUr,dUh = 0,0,0

        
        sampH=128
        samp_N, sampT, sampD = data_train.shape
  
        
        init_lyr=self.lyrs[0]
        delta=init_lyr.delta_lyr

        for t in reversed(range(sampT)):
            x = data_train[:, t]
            if t > 0:
                h_t_1 = h_t[:, t - 1]
            else:
                h_t_1 = np.zeros((samp_N, sampH))
            

            dz =  (h_t_t[:, t] - h_t_1) * init_lyr.act_derv(z_t[:, t],"sigmoid") *delta 
            dh_t_t =  z_t[:, t] * init_lyr.act_derv(h_t_t[:, t],"tanh")*delta 
            dr = dh_t_t.dot(init_lyr.Uh.T) * init_lyr.act_derv(r_t[:, t],"sigmoid")* h_t_1
            
            dWz += np.concatenate((x, -np.ones((samp_N, 1))), axis=1).T.dot(dz)
            dWh += np.concatenate((x, -np.ones((samp_N, 1))), axis=1).T.dot(dh_t_t)
            dWr +=np.concatenate((x, -np.ones((samp_N, 1))), axis=1).T.dot(dr)
            
            dUz += h_t_1.T.dot(dz)
            dUh += h_t_1.T.dot(dh_t_t)      
            dUr += h_t_1.T.dot(dr)                  
                              
                              
            # update the  gradients
          
            
            d9 =  (1 - z_t[:, t])*delta 
            d11 = dz.dot(init_lyr.Uz.T)
            d13 = dh_t_t.dot(init_lyr.Uh.T) * (r_t[:, t] + h_t_1 * init_lyr.act_derv(r_t[:, t],"sigmoid").dot(init_lyr.Ur.T))                  
                  
            delta = d9+d11+d13

        for i in range(len(self.lyrs)):
            lyr = self.lyrs[i]
            if(i == 0): 
                
                up_Wz,up_Wr,up_Wh=np.array([dWz,dWr,dWh])*l_rate/size_batch
                up_Uz,up_Ur,up_Uh=np.array([dUz,dUr,dUh])*l_rate/size_batch
                
                lyr.Wz+= up_Wz + lyr.prevU_Wz*momentum
                lyr.Uz+= up_Uz + lyr.prevU_Uz*momentum  
                lyr.Wr+= up_Wr + lyr.prevU_Wr*momentum 
                lyr.Ur+= up_Ur + lyr.prevU_Ur*momentum              
                lyr.Wh+= up_Wh + lyr.prevU_Wh*momentum  
                lyr.Uh+= up_Uh + lyr.prevU_Uh*momentum           

                
                lyr.prevU_Wz ,lyr.prevU_Wr,lyr.prevU_Wh=np.array([up_Wz,up_Wr,up_Wh])
                lyr.prevU_Uz ,lyr.prevU_Ur,lyr.prevU_Uh=np.array([up_Uz,up_Ur,up_Uh])
             
            else:      
                samp_N = len(self.lyrs[i - 1].Last_Act)
                inp_temp=np.concatenate((self.lyrs[i - 1].Last_Act, -np.ones((samp_N, 1))), axis=1)   
                upd =  (inp_temp.T.dot(lyr.delta_lyr))*l_rate/size_batch
                lyr.all_W+= upd + lyr.prevU*momentum
                lyr.prevU = upd      
                
    def train(self,l_rate,size_batch,data_train,lab_train, inp_test, lab_test, N_ep,momentum):
        cr_los_list = []   
        train_res=[]
        for ep in range(N_ep):
            print("ep:",ep)
            ind=np.random.permutation(len(data_train))
         
            data_train=data_train[ind]
            lab_train=lab_train[ind]
            batch_N = int(len(data_train)/size_batch) 
            for j in range(batch_N):
                train_data = data_train[j*size_batch:size_batch*(j+1)]
                train_labels = lab_train[j*size_batch:size_batch*(j+1)]
                self.back(l_rate,size_batch,train_data,train_labels,momentum)         
            _, out_val = self.Forward(inp_test)
            _, out_train = self.Forward(data_train)
            cr_los = np.sum(-np.log(out_val) * lab_test)/len(out_val)
            cr_los1 = np.sum(-np.log(out_train) * lab_train)/len(out_train)
            print('C-E Error of Validation', cr_los)
            print('C-E Error of Train', cr_los1)
            cr_los_list.append(cr_los)
            train_res.append(cr_los1)
        return cr_los_list, train_res
    
    def Predict(self,inps,realout):
        _,out = self.Forward(inps)
        out = out.argmax(axis=1)
        realout = realout.argmax(axis=1)
        return ((out == realout).mean()*100)
    
    
    def conf(self,inp,outp):
        _,pred= self.Forward(inp)
        pred = pred.argmax(axis=1)
        outp = outp.argmax(axis=1)
        q = len(np.unique(outp))
        conf=np.zeros((q,q))
        for p in range(len(outp)):
            conf[outp[p]][pred[p]] += 1
        return conf    
    


def q1():
    filename = 'data1.h5'
    data=h5py.File(filename, 'r')["data"][()]
    data_gray = 0.2126*data[:,0,:,:] + 0.7152*data[:,1,:,:] + 0.0722*data[:,2,:,:]
    mean_data = np.mean(data_gray, axis=(1,2))
   
    for k in range(len(mean_data)):
        data_gray[k,:,:] -= mean_data[k]
        
    std_data = np.std(data_gray)
    
    normalized = np.clip(data_gray, std_data*(-3),std_data*3) 
    
    normalization = lambda x:(x-x.min()) / (x.max()-x.min())
    
    data_nor= normalization(normalized)*0.8 + 0.1
    
    data_T=np.transpose(data,(0,2,3,1))
    random_sample=np.random.randint(0,len(data_nor),size=(200))
    
    row = 10
    col = 20
    fig=plt.figure(figsize=(20, 10), dpi= 100)
    for j in range(row*col):
        plt.subplot(row,col,j+1)
        rand_pic=random_sample[j]
        plt.imshow(data_T[rand_pic,:,:,:])
        plt.axis('off')
    plt.show()    

    fig=plt.figure(figsize=(20, 10), dpi= 100)
    for j in range(row*col):
        plt.subplot(row,col,j+1)
        rand_pic=random_sample[j]
        plt.imshow(data_nor[rand_pic,:,:],cmap='gray')
        plt.axis("off")
    plt.show()


    num_pixel=data_nor.shape[1]
    Lin = Lout = num_pixel**2
    Lhid = 64
    batch_size = 32
    epoch = 200
    lmb = 5e-4
    alpha = 0.85
    rho = 0.025
    eta = 0.075
    beta = 2
    
    params= (Lin, Lhid, lmb, beta, rho)
    
    data_solve=data_nor.reshape(data_nor.shape[0],data_nor.shape[1]**2)
    w,j=solver(data_solve, params, eta, alpha, epoch, batch_size)
    
    plot_weights(w)
    hid_list=[16,49,100]
    lmb_list=[0,2e-4,1e-3]
    j_list=[j]
    w_list=[w]
    for i in range(len(hid_list)):
        for j in range(len(lmb_list)):
            params= (Lin, hid_list[i], lmb_list[j], beta, rho)
            print("parameters: Lin={}, Lhid={}, lmb={}, beta={}, rho={} ".format(Lin, hid_list[i], lmb_list[j], beta, rho))
            w,j=solver(data_solve, params, eta, alpha, epoch, batch_size)
            j_list.append(j)
            w_list.append(w)
    for i in w_list[1:]:
        plot_weights(i)


def q2():
    filename="data2.h5"
    testx = h5py.File(filename, 'r')["testx"][()]
    traind = h5py.File(filename, 'r')["traind"][()]
    trainx = h5py.File(filename, 'r')["trainx"][()]
    vald = h5py.File(filename, 'r')["vald"][()]
    valx = h5py.File(filename, 'r')["valx"][()]
    words = h5py.File(filename, 'r')["words"][()]
    v_size = 250
    lr = 0.15
    moment = 0.85
    batch = 200
    epoch = 50
    


    valx1h = data_1h(valx, v_size)
    vald1h = label_1h(vald, v_size)
    
    P_1 = 256
    D_1 = 32
    
    model1 = Neural()
    model1.layer_add(Layer(D_1, v_size,0,0.25, 'emb'))
    model1.layer_add(Layer(3*D_1, P_1, 0,0.25, 'sigmoid'))
    model1.layer_add(Layer(P_1, v_size, 0,0.25,'softmax'))
    
    loss1 = model1.Train(lr,batch,trainx,traind,valx1h,vald1h,epoch,moment,v_size)
    
    P_2 = 128
    D_2 = 16
    
    model2 = Neural()
    model2.layer_add(Layer(D_2, v_size,0,0.25, 'emb'))
    model2.layer_add(Layer(3*D_2, P_2, 0,0.25, 'sigmoid'))
    model2.layer_add(Layer(P_2, v_size, 0,0.25,'softmax'))
    
    loss2 = model2.Train(lr,batch,trainx,traind,valx1h,vald1h,epoch,moment,v_size)
    
    
    P_3 = 64
    D_3 = 8
    
    model3 = Neural()
    model3.layer_add(Layer(D_3, v_size,0,0.25, 'emb'))
    model3.layer_add(Layer(3*D_3, P_3, 0,0.25, 'sigmoid'))
    model3.layer_add(Layer(P_3, v_size, 0,0.25,'softmax'))
    
    loss3 = model3.Train(lr,batch,trainx,traind,valx1h,vald1h,epoch,moment,v_size)
    
    plt.plot(loss1,label="D=32,P=256")
    plt.plot(loss2,label="D=16,P=128")
    plt.plot(loss3,label="D=8,P=64")
    plt.legend()
    plt.title('Cross Entropy Error Plots For Different (D,P) Values')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Cross Entropy Error')
    plt.show()
    
    
    rnd = np.random.permutation(len(testx))[0:5]
    
    samp = testx[rnd,:]

    
    samp1h = data_1h(samp, 250)
    
    preds1 = model1.predict(samp1h, 10)
    
    for i in range(5):
        print('Triagram {}: \n'.format(i+1))
        st=""
        for q in range(3):
            st+=str(words[samp[i,q]-1].decode("utf-8"))+" "
        
        print(st+"\n")
        print('Most Probable 10 Predictions are:\n ')
        w_list=[]
        for j in range(10):
            w_list.append(str(words[preds1[j,i]-1].decode("utf-8")))
        for j in w_list:
            print(j)
        print("\n")
    
    
    preds2 = model2.predict(samp1h, 10)
    
    for i in range(5):
        print('Triagram {}: \n'.format(i+1))
        st=""
        for q in range(3):
            st+=str(words[samp[i,q]-1].decode("utf-8"))+" "
        
        print(st+"\n")
        print('Most Probable 10 Predictions are:\n ')
        w_list=[]
        for j in range(10):
            w_list.append(str(words[preds2[j,i]-1].decode("utf-8")))
        for j in w_list:
            print(j)
        print("\n")
        
        
    preds3 = model3.predict(samp1h, 10)
    
    for i in range(5):
        print('Triagram {}: \n'.format(i+1))
        st=""
        for q in range(3):
            st+=str(words[samp[i,q]-1].decode("utf-8"))+" "
        
        print(st+"\n")
        print('Most Probable 10 Predictions are:\n ')
        w_list=[]
        for j in range(10):
            w_list.append(str(words[preds3[j,i]-1].decode("utf-8")))
        for j in w_list:
            print(j)
        print("\n")
def q3(t):
    file = h5py.File("data3.h5", "r") # get data
    
    dat_tra = np.array(file[list(file.keys())[0]])
    lab_tra = np.array(file[list(file.keys())[1]])
    dat_test = np.array(file[list(file.keys())[2]])
    lab_test = np.array(file[list(file.keys())[3]])
    
 
    
    neuron_n=128    
    batch = 32  
    epoch = 30
    alpha = 0.85 # momentum 
    eta = 0.03 #learning rate
    ind=np.random.permutation(len(dat_tra))
    dat_tra=dat_tra[ind]
    lab_tra=lab_tra[ind]
    
    size_v = int(len(dat_tra) / 10)
    dat_v=dat_tra[:size_v]
    lab_v=lab_tra[:size_v]
    dat_tra1=dat_tra[size_v:]
    lab_tra1=lab_tra[size_v:]   
    
    if t==1:
        model_rnn= RNN_(dat_tra1)
        model_rnn.lyr_add(lyr(3, neuron_n,'hyperbolic',1)) #3 val sensor
        model_rnn.lyr_add(lyr(neuron_n,70,'relu',1))
        model_rnn.lyr_add(lyr(70,30,'relu',1))
        model_rnn.lyr_add(lyr(30,6,'softmax',1))
        creL_rnn, lis_tra_rnn = model_rnn.train(eta,batch,dat_tra1,lab_tra1,dat_v,lab_v,epoch,alpha)
        
        plt.plot(creL_rnn)
        
        plt.xlabel('Number of Epochs')
        plt.ylabel('C-E Error ')
        plt.title('C-E Error of of Validation for RNN')
        plt.show()
        
        acc_test_rnn=model_rnn.Predict(dat_test,lab_test)
        acc_train_rnn=model_rnn.Predict(dat_tra1,lab_tra1)
        print("Test set Accuracy: "+str(acc_test_rnn)+"%")
        print("Train set Accuracy: "+str(acc_train_rnn)+"%")
        
        
        
        
        conf_test_rnn=model_rnn.conf(dat_test,lab_test) 
        sn.heatmap(conf_test_rnn, yticklabels=[1, 2, 3, 4, 5, 6], xticklabels=[1, 2, 3, 4, 5, 6],  cmap=sn.cm.rocket_r, fmt='g')
        plt.xlabel("Prediction")
        plt.title("Confusion Matrix of test set for rnn")
        plt.ylabel("Real")
        plt.show()
        
        conf_train_rnn=model_rnn.ConfusionMatrix(dat_tra1,lab_tra1) 
    
        sn.heatmap(conf_train_rnn, yticklabels=[1, 2, 3, 4, 5, 6], xticklabels=[1, 2, 3, 4, 5, 6],  cmap=sn.cm.rocket_r, fmt='g')
        plt.xlabel("Prediction")
        plt.title("Confusion Matrix of train set for rnn")
        plt.ylabel("Real")
        plt.show()

    elif t==2:
        model_lstm= LSTM_(dat_tra1)
        model_lstm.lyr_add(LSTM_lyr(131, neuron_n,1)) #3 val from sensor 128 prev
        model_lstm.lyr_add(lyr(neuron_n,70,'relu',1))
        model_lstm.lyr_add(lyr(70,30,'relu',1))
        model_lstm.lyr_add(lyr(30,6,'softmax',1))
        creL_lstm, lis_tra_lstm = model_lstm.train(eta,batch,dat_tra1,lab_tra1,dat_v,lab_v,epoch,alpha)
        
        plt.plot(creL_lstm)
        
        plt.xlabel('Number of Epochs')
        plt.ylabel('C-E Error ')
        plt.title('C-E Error ofValidation for LSTM')
        plt.show()
        
        acc_test_lstm=model_lstm.Predict(dat_test,lab_test)
        acc_train_lstm=model_lstm.Predict(dat_tra1,lab_tra1)
        print("Test set Accuracy: "+str(acc_test_lstm)+"%")
        print("Train set Accuracy: "+str(acc_train_lstm)+"%")
        
        
        
        
        conf_test_lstm=model_lstm.conf(dat_test,lab_test) 
        sn.heatmap(conf_test_lstm, yticklabels=[1, 2, 3, 4, 5, 6], xticklabels=[1, 2, 3, 4, 5, 6],  cmap=sn.cm.rocket_r, fmt='g')
        plt.xlabel("Prediction")
        plt.title("Confusion Matrix of test set for lstm")
        plt.ylabel("Real")
        plt.show()
        
        conf_train_lstm=model_lstm.ConfusionMatrix(dat_tra1,lab_tra1) 
        sn.heatmap(conf_train_lstm, yticklabels=[1, 2, 3, 4, 5, 6], xticklabels=[1, 2, 3, 4, 5, 6],  cmap=sn.cm.rocket_r, fmt='g')
        plt.xlabel("Prediction")
        plt.title("Confusion Matrix of train set for lstm")
        plt.ylabel("Real")
        plt.show()
    elif t==3:
        model_gru = GRU_(dat_tra1)
        model_gru.lyr_add(GRU_lyr(3, neuron_n,1)) 
        model_gru.lyr_add(lyr(neuron_n,70,'relu',1))
        model_gru.lyr_add(lyr(70,30,'relu',1))
        model_gru.lyr_add(lyr(30,6,'softmax',1))
        creL_gru, lis_tra_gru = model_gru.train(eta,batch,dat_tra1,lab_tra1,dat_v,lab_v,epoch,alpha)
        
        plt.plot(creL_gru)
        
        plt.xlabel('Number of Epochs')
        plt.ylabel('C-E Error ')
        plt.title('C-E Error ofValidation for GRU')
        plt.show()
        
        acc_test_gru=model_gru.Predict(dat_test,lab_test)
        acc_train_gru=model_gru.Predict(dat_tra1,lab_tra1)
        print("Test set Accuracy: "+str(acc_test_gru)+"%")
        print("Train set Accuracy: "+str(acc_train_gru)+"%")
        
        
        
        
        conf_test_gru=model_gru.conf(dat_test,lab_test) 
        sn.heatmap(conf_test_gru, yticklabels=[1, 2, 3, 4, 5, 6], xticklabels=[1, 2, 3, 4, 5, 6],  cmap=sn.cm.rocket_r, fmt='g')
        plt.xlabel("Prediction")
        plt.title("Confusion Matrix of test set for gru")
        plt.ylabel("Real")
        plt.show()
        
        conf_train_gru=model_gru.ConfusionMatrix(dat_tra1,lab_tra1) 
    
        sn.heatmap(conf_train_gru, yticklabels=[1, 2, 3, 4, 5, 6], xticklabels=[1, 2, 3, 4, 5, 6],  cmap=sn.cm.rocket_r, fmt='g')
        plt.xlabel("Prediction")
        plt.title("Confusion Matrix of train set for gru")
        plt.ylabel("Real")
        plt.show()

## in order to run the first question please use q1() function
## in order to run the second question please use q2() function
##in order to run the third question please use the q3(t) function, such that t can only take 1,2 or 3
## those t values represent the part of the third question
## Thus, in order to run RNN code use q3(1)
## in order to run LSTM code use q3(2)
## in order to run GRU code use q3(3)
