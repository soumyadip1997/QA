import torch.optim as optim
import torch.nn as nn

from QA.data_import import get_dataloader,data1
import matplotlib.pyplot as plt
#from QA.temp_network import GNN
import torch
def test(GNN,get_dataloader,saved_model):
    model=GNN
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model=model.to(device)
    
    model.load_state_dict(torch.load(saved_model))
    result_list=[]
    target_list=[]
    
    for i,batch in enumerate(get_dataloader):

            print(f'--->{i}')
            one_hot_atom=batch[0]
            one_hot_res=batch[1]
            neigh_same_res=batch[2]
            neigh_diff_res=batch[3]
            target=batch[4]
            mini_batch=[]
            for p in range(len(batch[0])):
            
              result=model([one_hot_atom[p].to(device),one_hot_res[p].to(device),neigh_same_res[p],neigh_diff_res[p]])
              mini_batch.append(result)

              # explain the flatten 
              #Result has a dimension of [N,1] and so we are trying to convert it to  a 1D tensor inorder to plot it
              #So we are flattening the result and also the target
              result_list.append(torch.flatten(result.detach().cpu()).numpy())
            target_list.append(torch.flatten(target.detach().cpu()).numpy())
              #print(result_list)
            
    #result_list now consist of  a number o numpy arrays like [[<array 1>],[<array 2>], ..] but we need to flatten it 
    # to plot it in a graph like [<array 1 contents>,<array 2 contents>,.... ]
    result_list=np.array(result_list).flatten()

    target_list=np.array(target_list).flatten()
    plt.plot(target_list,result_list,".")
    x = np.linspace(0,1,100)
    y=x
    plt.plot(x,y)
    plt.xlabel("Actual scores")
    plt.ylabel("Predicted Scores")

