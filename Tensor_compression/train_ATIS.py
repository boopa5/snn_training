import torch
import torch.nn as nn
import argparse
import time

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.utils.data

from torch.autograd import Variable
from torch.utils.data import Dataset


import numpy as np
import sklearn.metrics

from tensor_layers.layers import TensorizedEmbedding, TensorizedLinear_module

from meta_optimizer.optimizee import MetaModel
from meta_optimizer.nn_optimizer.foopt import FOOptimizer


parser = argparse.ArgumentParser()


parser.add_argument('-save_model', default=None)
parser.add_argument('-batch_size',type=int, default=32)
parser.add_argument('-max_length',type=int, default=32)
parser.add_argument('-use_cuda_graph',type=int, default=0)



opt = parser.parse_args()
opt.use_cuda_graph = (opt.use_cuda_graph==1)


torch.manual_seed(0)




# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True

def get_kl_loss(model, epoch, tensor_blocks = None, no_kl_epochs=5, warmup_epochs=10):

    kl_loss = 0.0
    n_total = 1
    for layer in model.modules():
        if hasattr(layer, "tensor"):
            kl,n= layer.tensor.get_kl_divergence_to_prior()
            kl_loss += kl
            n_total += n

    kl_mult = 1e-4 * torch.clamp(
                            torch.tensor((
                                (epoch - no_kl_epochs) / warmup_epochs)), 0.0, 1.0)
    """
    print("KL loss ",kl_loss.item())
    print("KL Mult ",kl_mult.item())
    """
    return kl_loss/n_total

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}



from torch.nn.utils.rnn import pad_sequence

def collate_fn_custom(batch):
    src_batch, sim_batch = [], []
    src_attn_batch = []
    slot_batch = []
    seg_batch = []
    for similar,seq,slot,seg in batch:
        
        seq = np.insert(seq,0,0)
        seg.append(1)
        
      
        src_batch.append(torch.tensor(seq))
        src_attn_batch.append(torch.tensor([1]*len(src_batch[-1])))
        sim_batch.append(similar)

        seg_batch.append(torch.tensor([1]*len(src_batch[-1])))
        slot_batch.append(torch.tensor(slot)-1) #ATIS 

    src_batch = pad_sequence(src_batch, padding_value=0)
    src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
    seg_batch = pad_sequence(seg_batch, padding_value=0)
    slot_batch = pad_sequence(slot_batch, padding_value=-100)


    return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(slot_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)


def collate_fn_cudagraph(batch):
    src_batch, sim_batch = [], []
    src_attn_batch = []
    slot_batch = []
    seg_batch = []
    
    max_len = opt.max_length
    for similar,seq,slot,seg in batch:
        
        if len(seq)>max_len-1:
            seq = seq[:max_len-1]
            slot = slot[:max_len-2]
        
        seq = np.insert(seq,0,0)
        
        

        
        src_batch.append(torch.tensor(list(seq)+[-99]*(max_len-len(seq))))
        src_attn_batch.append(torch.tensor([1]*len(src_batch[-1])+[0]*(max_len-len(src_batch[-1]))))
        sim_batch.append(similar)

        seg_batch.append(torch.tensor([1]*len(src_batch[-1])+[0]*(max_len-len(src_batch[-1]))))
        
        slot_batch.append(torch.tensor(list(slot)+(max_len-1-len(slot))*[-99])-1) #ATIS 

    src_batch = pad_sequence(src_batch, padding_value=0)
    src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
    seg_batch = pad_sequence(seg_batch, padding_value=0)
    slot_batch = pad_sequence(slot_batch, padding_value=-100)


    return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(slot_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)

#========= Loading Dataset =========#


from torch.utils.data import DataLoader


train_iter = torch.load('./examples/ATIS/data/ATIS_train.pt')
if opt.use_cuda_graph:
    training_data = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn_cudagraph, shuffle=True, drop_last=True)
else:
    training_data = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=True)


val_iter = torch.load('./examples/ATIS/data/ATIS_valid.pt')
validation_data = DataLoader(val_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=False)

test_iter = torch.load('./examples/ATIS/data/ATIS_test.pt')
test_data = DataLoader(test_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=False)






device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#Prepare model

from tensor_layers.utils import config_class
from tensor_layers.Transformer_tensor import Transformer_classification,Transformer_classification_SLU

# transformer config
D = {
    'n_layers': 2,
    'vocab_size': 1000,
    'n_position': 512,
    'd_model':768,
    'd_hid':768*4,
    'n_head':12,
    'tensorized':True,
    'dropout': 0.1,
    'embedding': None,
    'classification': None,
    'pff': {},
    'attn': {}
    }

set_scale_factors = False

# emb_shape = [[10,10,10],[12,8,8]]
emb_shape = [[5,5,5,8],[6,4,8,4]]
emb_rank = 30

r = 10
attn_shape = [12,8,8,8,8,12]
pff_shape = [[12,8,8,12,16,16],[16,16,12,8,8,12]]
attn_rank = r
pff_rank = [r,r]

classification_shape = [12,8,8,8,8,12]
classification_rank = 20


config_model =config_class(**D)

config_model.pff[0] = config_class(shape=pff_shape[0],ranks=pff_rank[0],set_scale_factors=set_scale_factors)
config_model.pff[1] = config_class(shape=pff_shape[1],ranks=pff_rank[1],set_scale_factors=set_scale_factors)


config_attn_sublayers = config_class(shape=attn_shape,ranks=attn_rank,set_scale_factors=set_scale_factors)
for key in ['q','k','v','fc']:
    config_model.attn[key] = config_attn_sublayers


config_model.embedding = config_class(shape=emb_shape,ranks=emb_rank,set_scale_factors=set_scale_factors)


num_class = 22
slot_num = 121

config_classification = config_class(d_model=D['d_model'],tensorized=D['tensorized'],num_class=num_class,dropout=D['dropout'],shape=classification_shape,ranks=classification_rank,set_scale_factors=set_scale_factors)

config_slot = config_class(d_model=D['d_model'],tensorized=D['tensorized'],num_class=slot_num,dropout=D['dropout'],shape=classification_shape,ranks=classification_rank,set_scale_factors=set_scale_factors)

config_forward = config_class(prune_mask=True,threshold=10,quantized=0)
transformer = Transformer_classification_SLU(config_model,config_classification,config_slot)


transformer.to(device)


precondition = False



model = transformer



def sample_model():
    # transformer config
    D = {
        'n_layers': 2,
        'vocab_size': 1000,
        'n_position': 512,
        'd_model':768,
        'd_hid':768*4,
        'n_head':12,
        'tensorized':True,
        'dropout': 0.1,
        'embedding': None,
        'classification': None,
        'pff': {},
        'attn': {}
        }

    set_scale_factors = False

    # emb_shape = [[10,10,10],[12,8,8]]
    emb_shape = [[5,5,5,8],[6,4,8,4]]
    emb_rank = 30

    r = 10
    attn_shape = [12,8,8,8,8,12]
    pff_shape = [[12,8,8,12,16,16],[16,16,12,8,8,12]]
    attn_rank = r
    pff_rank = [r,r]

    classification_shape = [12,8,8,8,8,12]
    classification_rank = 20


    config_model =config_class(**D)

    config_model.pff[0] = config_class(shape=pff_shape[0],ranks=pff_rank[0],set_scale_factors=set_scale_factors)
    config_model.pff[1] = config_class(shape=pff_shape[1],ranks=pff_rank[1],set_scale_factors=set_scale_factors)


    config_attn_sublayers = config_class(shape=attn_shape,ranks=attn_rank,set_scale_factors=set_scale_factors)
    for key in ['q','k','v','fc']:
        config_model.attn[key] = config_attn_sublayers


    config_model.embedding = config_class(shape=emb_shape,ranks=emb_rank,set_scale_factors=set_scale_factors)


    num_class = 22
    slot_num = 121

    config_classification = config_class(d_model=D['d_model'],tensorized=D['tensorized'],num_class=num_class,dropout=D['dropout'],shape=classification_shape,ranks=classification_rank,set_scale_factors=set_scale_factors)

    config_slot = config_class(d_model=D['d_model'],tensorized=D['tensorized'],num_class=slot_num,dropout=D['dropout'],shape=classification_shape,ranks=classification_rank,set_scale_factors=set_scale_factors)

    # config_forward = config_class(prune_mask=True,threshold=10,quantized=0)
    transformer = Transformer_classification_SLU(config_model,config_classification,config_slot)


    transformer.to(device)

    # precondition = False
    return transformer


def main():


    epochs = 100

    # config_forward = config_class(prune_mask=False,threshold=10,quantized=1)
    # D_scale = {
    # 'scale_w': 2**(-3),
    # 'scale_intermediate': 2**(-5),
    # 'scale_dy': 1.0,
    # 'scale_x': 2**(-3),
    # 'scale_out': 1.0}

    D_scale = {
    'scale_w': 1.0,
    'scale_intermediate': 1.0,
    'scale_dy': 1.0,
    'scale_x': 2**(-1),
    'scale_out': 1.0,
    'scale_input': 1.0
    }

    D_emb_scale = {
    'scale_w': 1.0
    }

    
    



    D_fwd ={
    'prune_mask':False,
    'threshold':1e-2,
    'quantized':0,
    'rep': 'FLOAT',
    'rounding': 'nearest',
    'bit_input': [1,5,2],
    'bit_factors': [1,5,2],
    'bit_intermediate': [1,5,2],
    'bit_out': [1,5,2],
    'emb_quantized': 0,
    'emb_bit_factors': [1,5,2],
    'emb_rep': 'FLOAT',
    'emb_rounding': 'nearest'
    }
    config_forward = config_class(**D_fwd)
    
    if config_forward.quantized!=0:
        for mod in model.modules():
            if type(mod) == TensorizedLinear_module:
                mod.set_scale_factors(**D_scale)
            elif type(mod) == TensorizedEmbedding:
                mod.set_scale_factors(**D_emb_scale)


    if config_model.tensorized == True:
         
        lr = 5e-4

        par = list(model.parameters())
        optimizer = optim.Adam(par, betas=(0.9, 0.98), eps=1e-06, lr = lr)


    else:
        optimizer = optim.Adam(
                        filter(lambda x: x.requires_grad, model.parameters()),
                        betas=(0.9, 0.98), eps=1e-06, lr = 1e-4)

    valid_acc_all = [-1]
    train_result = []
    test_result = []

    optimizee = sample_model().to(device)

    # for name, module in optimizee.named_parameters():
    #     print(name)
        # if modules_map[module]
        # print(module._get_name())
        # if module._get_name() == 'TensorTrain':
        #     print(module)
        #     print(module._parameters)
        #     print("-"*50)
        #     for child in module.children():
        #         print(child)
        #         print("-"*50)
        # print("-"*50)

    meta_optimizer_config = {
        'model': MetaModel(optimizee),
        'num_layers': 1,
        'hidden_size': 10,
        'device': device,
        'partition_func': lambda p_name: 1 if 'factors' in p_name else 0
    }

    meta_optimizer = FOOptimizer(**meta_optimizer_config).to(device)
    meta_opt_opt = optim.Adam(meta_optimizer.parameters())

# def train_meta_optimizer(meta_optimizer, training_data, num_epochs, updates_per_epoch, optimizer, optimizer_steps, truncated_bptt_step, device='cuda', config_forward=None):
    train_meta_optimizer_config = {
        'meta_optimizer': meta_optimizer,
        'training_data': training_data,
        'num_epochs': 10,
        'updates_per_epoch': 100,
        'optimizer': meta_opt_opt,
        'optimizer_steps': 100,
        'truncated_bptt_step': 20,
        'device': device,
        'config_forward': config_forward
    }


    meta_optimizer = train_meta_optimizer(**train_meta_optimizer_config)
    
    if opt.use_cuda_graph:
        # transformer = torch.compile(model)
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        static_tensors = build_graph(transformer,optimizer,loss_fn,opt.batch_size,opt.max_length,device)
    
    for epoch in range(epochs):
        start = time.time()

        if opt.use_cuda_graph:
            train_loss, train_accu, train_slot_accu = run_graph(training_data,*static_tensors)
        else:
            # train_loss, train_accu, train_slot_accu = train_epoch_bylayer(transformer, training_data, optimizer,device=device,step=epoch,config_forward=config_forward)
            train_loss, train_accu, train_slot_accu = meta_train_epoch_bylayer(transformer, training_data, meta_optimizer,device=device,step=epoch,config_forward=config_forward)

        # train_loss, train_accu = 0,0

        start_val = time.time()

        valid_loss, valid_accu, valid_slot_accu = eval_epoch(transformer, validation_data, device,config_forward=config_forward)

        test_loss, test_accu, test_slot_accu = eval_epoch(transformer, test_data, device,config_forward=config_forward)

        train_result += [train_loss.cpu().to(torch.float32),train_accu.cpu().to(torch.float32),train_slot_accu.cpu().to(torch.float32)]
        test_result += [test_loss,test_accu.cpu().to(torch.float32),test_slot_accu]


        end = time.time()



        train_loss_new = 0
        
        print('')
        print('epoch = ', epoch)
        
        print('  - (Training)   loss: {loss: 8.5f}, loss_hard: {loss_new: 8.5f}, accuracy: {accu:3.3f} %, slot accuracy: {slot_accu:3.3f},'\
            'elapse: {elapse:3.3f} min'.format(
                loss=train_loss, loss_new=train_loss_new, accu=100*train_accu, slot_accu = 100*train_slot_accu,
                elapse=(start_val-start)/60))
       
        
        print('  - (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, slot accuracy: {slot_accu:3.3f},'\
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss, accu=100*valid_accu,slot_accu = 100*valid_slot_accu,
                    elapse=(end-start_val)/60))

        print('  - (Test) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, slot accuracy: {slot_accu:3.3f},'\
                'elapse: {elapse:3.3f} min'.format(
                    loss=test_loss, accu=100*test_accu,slot_accu = 100*test_slot_accu,
                    elapse=(end-start_val)/60))
        
        full_model_name = opt.save_model + '.chkpt'
        torch.save(transformer.state_dict(),full_model_name)

        if max(valid_acc_all)<valid_accu:
            best_model_name = opt.save_model + '_best' + '.chkpt'
            torch.save(transformer.state_dict(),best_model_name)
        valid_acc_all.append(valid_accu)
    PATH_np = opt.save_model + '.npy'
    np.save(PATH_np,np.array([train_result,test_result]))


def train_meta_optimizer(meta_optimizer, training_data, num_epochs, updates_per_epoch, optimizer, optimizer_steps, truncated_bptt_step, device='cuda', config_forward=None):
    ''' Trains a meta optimizer on a given model '''

    print(meta_optimizer)
    # setup meta_optimizer in main

    Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    for epoch in range(num_epochs):
        decrease_in_loss = 0.0
        final_loss = 0.0
        train_iter = iter(training_data)
        for i in range(updates_per_epoch):

            # Sample a new model
            optimizee = sample_model().to(device)

            target, w1, slot_label,attn,seg = map(lambda x: Variable(x.to(device)), next(train_iter))

            pred,pred_slot = optimizee(w1,mask=attn,seg=seg,config_forward=config_forward)
            pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
            slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)

            loss_MLM =  Loss(pred_slot, slot_label)
            initial_loss = Loss(pred,target)  + loss_MLM

            for k in tqdm(range(optimizer_steps // truncated_bptt_step)):
                # TBPTT
                meta_optimizer.reset_state(keep_states=k>0, model=optimizee)
                loss_sum = 0
                prev_loss = torch.zeros(1).to(device)

                for j in range(truncated_bptt_step):
                    target, w1, slot_label,attn,seg = map(lambda x: Variable(x.to(device)), next(train_iter))

                    pred,pred_slot = optimizee(w1,mask=attn,seg=seg,config_forward=config_forward)
                    pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
                    slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)

                    loss_MLM =  Loss(pred_slot, slot_label)
                    loss = Loss(pred,target)  + loss_MLM

                    optimizee.zero_grad()
                    loss.backward()


                    # Perform a meta update using gradients from model
                    # and return the current meta model saved in the optimizer
                    meta_model = meta_optimizer.meta_update(optimizee)

                    # Compute a loss for a step the meta optimizer
                    pred, pred_slot = meta_model(w1,mask=attn,seg=seg,config_forward=config_forward)
                    pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)

                    loss_MLM = Loss(pred_slot, slot_label)
                    loss = Loss(pred,target)  + loss_MLM
                    loss_sum += (loss - Variable(prev_loss))
                    prev_loss = loss.data


                meta_optimizer.zero_grad()
                loss_sum.backward()
                for param in meta_optimizer.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

                # decrease_in_loss += loss.data[0] / initial_loss.data[0]
                # final_loss += loss.data[0]
                decrease_in_loss += loss.item() / initial_loss.item()
                final_loss += loss.item()
            
            valid_loss, valid_accu, valid_slot_accu = eval_epoch(optimizee, validation_data, device,config_forward=config_forward)
            print('  - (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, slot accuracy: {slot_accu:3.3f},'.format(
                    loss=valid_loss, accu=100*valid_accu,slot_accu = 100*valid_slot_accu))


            print("Epoch: {}, final loss {}, average final/initial loss ratio: {}".format(epoch, final_loss / updates_per_epoch,
                                                                       decrease_in_loss / updates_per_epoch))
    
    return meta_optimizer

def meta_train_epoch_bylayer(model, training_data, meta_optimizer: nn.Module, device='cuda',step=1, config_forward=None):
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    cos_total = 0
    attn_total = 0

    slot_total = 0
    slot_correct = 0

    count = 0

    meta_optimizer.reset_state(keep_states=False, model=model)
    meta_optimizer.eval()

    
    Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        target, w1, slot_label,attn,seg= map(lambda x: x.to(device), batch)


        model.zero_grad()


        pred,pred_slot = model(w1,mask=attn,seg=seg,config_forward=config_forward)
        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
        slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)            


        loss_MLM =  Loss(pred_slot, slot_label)
        loss = Loss(pred,target)  + loss_MLM

        if type(meta_optimizer) == FOOptimizer:
            loss.backward()
            meta_model = meta_optimizer.meta_update(model)


        # note keeping
        total_loss += loss.detach()*pred.shape[0]
        n_word_total += pred.shape[0]
        n_word_correct += torch.sum(torch.argmax(pred.detach(),dim=1)==target)

        slot_total += torch.sum(slot_label>=0).detach()
        slot_correct += torch.sum((torch.argmax(pred_slot.detach(),dim=-1)==slot_label)*(slot_label>=0)).detach()

        count += 1
        

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    accuracy_slot = slot_correct/slot_total

    return loss_per_word, accuracy, accuracy_slot

def train_epoch_bylayer(model, training_data, optimizer,device='cuda',step=1,config_forward=None):
    ''' Epoch operation in training phase'''


    model.train()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    cos_total = 0
    attn_total = 0

    slot_total = 0
    slot_correct = 0

    count = 0

    max_memory = 0

    Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        target, w1, slot_label,attn,seg= map(lambda x: x.to(device), batch)


        optimizer.zero_grad()

      
        pred,pred_slot = model(w1,mask=attn,seg=seg,config_forward=config_forward)
        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
        slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)
        



        loss_MLM =  Loss(pred_slot, slot_label)
        loss = Loss(pred,target)  + loss_MLM

        loss.backward()

        ####gradient based optimizer        
        optimizer.step() 

        # note keeping
        total_loss += loss.detach()*pred.shape[0]
        n_word_total += pred.shape[0]
        n_word_correct += torch.sum(torch.argmax(pred.detach(),dim=1)==target)

        slot_total += torch.sum(slot_label>=0).detach()
        slot_correct += torch.sum((torch.argmax(pred_slot.detach(),dim=-1)==slot_label)*(slot_label>=0)).detach()

        count += 1
        

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    accuracy_slot = slot_correct/slot_total

    return loss_per_word, accuracy, accuracy_slot



def eval_epoch(model, validation_data, device,config_forward=None):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    slot_total = 0
    slot_correct = 0

    slot_label_total = torch.tensor([0])
    pred_label_total = torch.tensor([0])

    slot_correct_all = []
    slot_pred_all = []

    Loss = nn.CrossEntropyLoss(label_smoothing=0.1,ignore_index=- 100)

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            target, w1, slot_label,attn,seg= map(lambda x: x.to(device), batch)


            # attn = None
            pred,pred_slot = model(w1,mask=attn,seg=seg,config_forward=config_forward)
            pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)

            slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)


            loss_MLM =  Loss(pred_slot, slot_label)


            loss = loss_MLM +  Loss(pred,target) 


        # print(loss)
            total_loss += loss.item()



            n_word_total += pred.shape[0]
            n_word_correct += torch.sum(torch.argmax(pred,dim=1)==target)

            slot_total += torch.sum(slot_label>=0).detach()
            slot_correct += torch.sum((torch.argmax(pred_slot.detach(),dim=-1)==slot_label)*(slot_label>=0)).detach()

            pred_label = (torch.argmax(pred_slot.detach(),dim=-1))[slot_label>=0]
            # pred_slot[slot_label>=0]
            slot_label = slot_label[slot_label>=0]

            # print(slot_label.shape)
            # print(pred_label.shape)


            slot_label_total=torch.cat((slot_label_total, slot_label.to('cpu')))
            pred_label_total=torch.cat((pred_label_total, pred_label.to('cpu')))

            slot_correct_all.append(slot_label.tolist())
            slot_pred_all.append(pred_label.tolist())



    f1_score = sklearn.metrics.f1_score(slot_label_total[1:].numpy(),pred_label_total[1:].numpy(),average='micro') # old is average = 'weighted'



    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    accuracy_slot = slot_correct/slot_total
    # accuracy_slot
    return loss_per_word, accuracy, f1_score


def build_graph(model,optimizer,loss_fn,batch_size,max_len,device):
    for U in optimizer.param_groups:
        U['capturable'] = True
    
    static_input = torch.randint(0,100,(batch_size, max_len), device=device)
    static_target_intent = torch.randint(0,2,(batch_size,), device=device)
    static_target_slot = torch.randint(0,100,(batch_size*(max_len-1),), device=device)
    
    static_attn = torch.randint(0,2,(batch_size,max_len), device=device)
    static_seg = torch.randint(0,2,(batch_size, max_len), device=device)
    
    s = torch.cuda.Stream(device)
    s.wait_stream(torch.cuda.current_stream(device))
    
    with torch.cuda.stream(s):
        for i in range(12):
            optimizer.zero_grad(set_to_none=True)
            y_pred,y_pred_slot_tmp = model(static_input,mask=static_attn,seg=static_seg)
            y_pred_slot = torch.flatten(y_pred_slot_tmp,start_dim=0, end_dim=1)
            loss = loss_fn(y_pred, static_target_intent)  + loss_fn(y_pred_slot,static_target_slot)
            loss.backward()
            optimizer.step()
    
    torch.cuda.current_stream(device).wait_stream(s)
    g = torch.cuda.CUDAGraph()

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.graph(g,stream=s):
        static_y_pred, temp = model(static_input,mask=static_attn,seg=static_seg)
        static_y_pred_slot = torch.flatten(temp,start_dim=0, end_dim=1)
        static_loss = loss_fn(static_y_pred, static_target_intent)  + loss_fn(static_y_pred_slot,static_target_slot)
        
        static_loss.backward()
        optimizer.step()

    return g,static_input,static_target_intent,static_target_slot,static_attn,static_seg,static_y_pred,static_y_pred_slot,static_loss

def run_graph(training_data,g,static_input,static_target_intent,static_target_slot,static_attn,static_seg,static_y_pred,static_y_pred_slot,static_loss):
    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    cos_total = 0
    attn_total = 0

    slot_total = 0
    slot_correct = 0

    count = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        target, w1, slot_label,attn,seg= batch
        
        slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)
        

        
        static_input.copy_(w1)
        static_target_slot.copy_(slot_label)
        static_target_intent.copy_(target)
        
        static_attn.copy_(attn)
        static_seg.copy_(seg)

        g.replay()

    
        # note keeping
        total_loss += static_loss.detach()*static_y_pred.shape[0]
        n_word_total += static_y_pred.shape[0]
        n_word_correct += torch.sum(torch.argmax(static_y_pred.detach(),dim=1)==static_target_intent.detach())

        slot_total += torch.sum(slot_label>=0).detach()
        slot_correct += torch.sum((torch.argmax(static_y_pred_slot.detach(),dim=-1)==static_target_slot.detach())*(static_target_slot.detach()>=0)).detach()

        count += 1
        

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    accuracy_slot = slot_correct/slot_total

    return loss_per_word, accuracy, accuracy_slot

if __name__ == '__main__':

    main()
