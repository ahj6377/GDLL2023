import argparse

parser = argparse.ArgumentParser(description='Layer Depedent Sampler')

parser.add_argument('--dataset', type=str, default='data',
                    help='Dataset name')
parser.add_argument('--n_hid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 2,
                    help='Number of Epoch')
parser.add_argument('--n_pool', type=int, default= 10,
                    help='Number of Pool')
parser.add_argument('--n_batch', type=int, default= 10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                    help='Number of iteration for a batch')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Number of batches when accuracy dont increase')
parser.add_argument('--n_samp', type=int, default=64,
                    help='Number of sampled nodes per layer')
parser.add_argument('--cuda', type=int, default=0,
                    help='GPU ID')



args = parser.parse_args()
