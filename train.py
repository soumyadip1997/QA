import torch.optim as optim
import torch.nn as nn

from QA.data_import import get_dataloader,data1
#from QA.temp_network import GNN
import torch
import matplotlib.pyplot as plt
def train(GNN,dataloader):    
    model=GNN
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model=model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss=nn.MSELoss()
    loss_list=[]
    
    for epoch in range(2):
        loss_per_epoch=0
        print(epoch)
        for i,batch in enumerate(dataloader):
            optimizer.zero_grad()
            mini_batch=[]
            one_hot_atom,one_hot_res,neigh_same_res,neigh_diff_res,target=batch
            for p in range(len(batch[0])):
            
              result=model([one_hot_atom[p].to(device),one_hot_res[p].to(device),neigh_same_res[p].to(device).long(),neigh_diff_res[p].to(device).long()])
              mini_batch.append(result)
              
            output = torch.stack(mini_batch)
            Loss=loss(output,(target.reshape(-1,1)).to(device))
            Loss.backward()
            optimizer.step()
            print(f'L2 Loss={Loss} for batch number {i}, for epoch = {epoch} ')
            loss_per_epoch+=Loss.item()
          
        loss_list.append(loss_per_epoch)
        
        print(f'Loss for epoch ={epoch} is {loss_per_epoch}')
    torch.save(model.state_dict(),'modelGCNL2_Global_Basic1.ckpt')
    plt.plot([(p+1) for p in range(len(loss_list))],loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
