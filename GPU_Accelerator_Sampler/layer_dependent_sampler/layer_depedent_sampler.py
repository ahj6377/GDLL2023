import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from lay_dep_sampler_model import *
import torch
import numpy as np
from sklearn.metrics import f1_score
import scipy.sparse as sp
from sampler_utils import get_adjacency_matrix, load_data,matrix_row_normalize, scipy_sparse_matrix_to_torch_sparse_matrix, mfgs_to_device
import scipy
import multiprocessing as mp
from sampler_arguments import args

if args.cuda != -1:
    device = torch.device("cuda:" + str(args.cuda))
else:
    device = torch.device("cpu")

def lay_dep_sampler(seeds, batch, fanout, n_nodes, row_normalize_matrix, depth):
    np.random.seed(seeds)
    prev_nodes_list = batch
    adjacency_list  = []
    for d in range(depth):
        Q = row_normalize_matrix[prev_nodes_list , :]
        prob_i = np.array(np.sum(Q.multiply(Q), axis=0))[0]
        prob = prob_i / np.sum(prob_i)
        s_num = np.min([np.sum(prob > 0), fanout[d]])
        next_nodes_list = np.random.choice(n_nodes, s_num, p = prob, replace = False)
        next_nodes_list = np.unique(np.concatenate((next_nodes_list, batch)))
        adjacency = Q[: , next_nodes_list].multiply(1/prob[next_nodes_list])
        adjacency_list += [scipy_sparse_matrix_to_torch_sparse_matrix(matrix_row_normalize(adjacency))]
        prev_nodes_list = next_nodes_list
    adjacency_list.reverse()
    return adjacency_list, prev_nodes_list, batch


def data_preparation(pool, sampler, procs_IDs, train_nodes, val_nodes, fanout, num_nodes, lap_matrix, depth):
    jobs = []
    for _ in procs_IDs:
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch = train_nodes[idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch, fanout, num_nodes, lap_matrix, depth))
        jobs.append(p)
    idx = torch.randperm(len(val_nodes))[:args.batch_size]
    batch = val_nodes[idx]
    p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch, fanout * 20, num_nodes, lap_matrix, depth))
    jobs.append(p)
    return jobs


edges, labels, features, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset)

adjacency_matrix = get_adjacency_matrix(edges, features.shape[0])

Q_r_matrix = matrix_row_normalize(adjacency_matrix + sp.eye(adjacency_matrix.shape[0]))
if type(features) == scipy.sparse.lil.lil_matrix:
    features = torch.FloatTensor(features.todense()).to(device)
else:
    features = torch.FloatTensor(features).to(device)
labels = torch.LongTensor(labels).to(device)

procs_IDS = np.arange(args.n_batch)

fanout = np.array([args.n_samp, args.n_samp, args.n_samp, args.n_samp, args.n_samp])

pool = mp.Pool(args.n_pool)
jobs = data_preparation(pool, lay_dep_sampler, procs_IDS, train_nodes, valid_nodes, fanout, len(features), Q_r_matrix, args.n_layers)

all_res = []
for iter in range(5):
    encd = GCN(n_feat= features.shape[1], n_hid=args.n_hid, layers=args.n_layers, drop_out= 0.2).to(device)
    model  = GCNModel(encd= encd, num_classes=num_classes, drop_out=0.5, inp = features.shape[1])
    model.to(device)
    optimizer = optim.Adam(filter(lambda p : p.requires_grad, model.parameters()))
    for epoch in np.arange(args.epoch_num):
        model.train()
        train_losses = []
        train_data = [job.get() for job in jobs[:-1]]
        valid_data = jobs[-1].get()
        pool.close()
        pool.join()
        pool = mp.Pool(args.n_pool)
        jobs = data_preparation(pool, lay_dep_sampler, procs_IDS, train_nodes, valid_nodes, fanout, len(features), Q_r_matrix, args.n_layers)
        for _iter in range(args.n_iters):
            for adjs, input_nodes, output_nodes in train_data:
                adjs = mfgs_to_device(adjs, device)
                optimizer.zero_grad()
                model.train()
                output = model.forward(features[input_nodes], adjs)
                loss_train = F.cross_entropy(output, labels[output_nodes])
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
                optimizer.step()
                train_losses += [loss_train.detach().tolist()]

        model.eval()
        adjs, input_nodes, output_nodes = valid_data
        adjs = mfgs_to_device(adjs, device)
        output = model.forward(features[input_nodes], adjs)
        loss_valid = F.cross_entropy(output, labels[output_nodes]).detach().tolist()
        valid_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
        print(("Epoch: %d Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f") % (epoch, np.average(train_losses), loss_valid, valid_f1))



