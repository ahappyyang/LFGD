import math
import torch.nn as nn
from model import SCDGN as Net
from utils import *
import argparse
from torch import optim
from model2 import my_model
import torch.nn.functional as F
import LDA_SLIC_SA
import warnings
from sklearn.cluster import KMeans
from functions import get_data,normalize,set_seed,spixel_to_pixel_labels,cluster_accuracy,cal_Neg,cal_norm
warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")#选择cpu或者GPU

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ICML')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs.')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stop.')
    parser.add_argument('--train', type=int, default=1, help='Train or not.')

    parser.add_argument('--cut', type=str, default=False, help='The type of degree.')  ##
    parser.add_argument('--type', type=str, default='sys', help='sys or rw')  ##
    parser.add_argument('--v_input', type=int, default=1, help='Degree of freedom of T distribution')  ##

    parser.add_argument('--imp_lr', type=float, default=1e-3, help='Learning rate of ICML.')  ##
    parser.add_argument('--exp_lr', type=float, default=1e-5, help='Learning rate of ICML.')  ##
    parser.add_argument('--imp_wd', type=float, default=1e-5, help='Weight decay of ICML.')
    parser.add_argument('--exp_wd', type=float, default=1e-5, help='Weight decay of ICML.')

    parser.add_argument('--knn', type=int, default=25, help='The K of KNN graph.')  ##
    parser.add_argument('--sigma', type=float, default=0.5, help='Weight parameters for knn.')  ##

    parser.add_argument("--hid_dim", type=int, default=512, help='Hidden layer dim.')  ##
    parser.add_argument('--time', type=float, default=20, help='End time of ODE integrator.')  ##
    parser.add_argument('--method', type=str, default='dopri5',
                        help="set the numerical solver: dopri5, euler, rk4, midpoint")
    parser.add_argument('--tol_scale', type=float, default=200, help='tol_scale .')  ##
    parser.add_argument('--add_source', type=str, default=True, help='Add source.')
    parser.add_argument('--dropout', type=float, default=0., help='drop rate.')  ##
    parser.add_argument('--n_layers', type=int, default=2, help='number of Linear.')  ##

    # Loss args
    parser.add_argument('--beta', type=float, default=1, help='Weight parameters for loss.')
    parser.add_argument('--gamma', type=float, default=1, help='Weight parameters for ICML.')

    parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")

    parser.add_argument('--dims', type=int, default=[500], help='Number of units in hidden layer 1.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate.')
    parser.add_argument('--sigma2', type=float, default=0.01, help='Sigma of gaussian distribution')

    parser.add_argument('--dataset', type=str, default='Salinas',
                        help='type of dataset.')  # 'Indian', 'Salinas', 'PaviaU'  'Houston','Trento'
    parser.add_argument('--superpixel_scale', type=int, default=150,
                        help="superpixel_scale")  # IP 100 sa  250  pu160  Tr900  HU100 pu160  Tr900  HU100

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'
    set_seed(args.seed)
    print(args)

    # Load data
    input, num_classes, y_true, gt_reshape, gt_hsi = get_data(args.dataset)
    # normalize data by band norm
    input_normalize = normalize(input)
    height, width, band = input_normalize.shape  # 145*145*200
    print("height={0},width={1},band={2}".format(height, width, band))
    input_numpy = np.array(input_normalize)


    ls = LDA_SLIC_SA.LDA_SLIC(input_numpy,gt_hsi, num_classes - 1)
    Q, S, A_SP, Edge_index, Edge_atter, Seg, A_ones = ls.simple_superpixel(scale=args.superpixel_scale)

    A=np.mat(A_ones)
    feat = torch.from_numpy(S).type(torch.FloatTensor)
    A_SP = torch.from_numpy(A_SP).type(torch.FloatTensor).to(device)


    ###########################################################################################
    n = 20
    F_group = torch.zeros([feat.shape[1] // n, feat.shape[0], n])
    for i in range(0, feat.shape[1], n):
        print(i)
        F_group[i // n] = feat[:, i:i + n]

    cos_sim = torch.nn.functional.cosine_similarity(F_group, F_group, dim=2)
    x_row_dup, x_col_dup = F_group[:, None, :, :], F_group[:, :, None, :]
    x_row_dup, x_col_dup = x_row_dup.expand(F_group.shape[0], F_group.shape[1], F_group.shape[1], F_group.shape[2]), x_col_dup.expand(
        F_group.shape[0], F_group.shape[1], F_group.shape[1], F_group.shape[2])
    x_cosine_similarity = torch.nn.functional.cosine_similarity(x_row_dup, x_col_dup, dim=-1)

    angle = torch.acos(torch.clamp(x_cosine_similarity, -1.0, 1.0))

    max = torch.max(angle)
    min = torch.min(angle)
    angle = max - angle

    k = torch.sum(angle, dim=1)

    min_vals, _ = torch.min(k, dim=1, keepdim=True)
    max_vals, _ = torch.max(k, dim=1, keepdim=True)

    scaled_k = ((k - min_vals) / (max_vals - min_vals)) + 1

    scaled_k = scaled_k.unsqueeze(2)

    a = torch.mul(scaled_k, F_group)

    a1 = a.transpose(0, 1)
    a2 = a1.reshape(a1.shape[0], -1)
    ###########################################################################################
    feat =  a2

    in_dim = feat.shape[1]
    args.N = N = feat.shape[0]
    norm_factor, edge_index, edge_weight, adj_norm, knn, Lap = cal_norm(A, args, feat)
    Lap_Neg = cal_Neg(adj_norm, knn, args)
    feat = feat.to(args.device)

    
    # Initial
    model = Net(N, edge_index, edge_weight, args).to(args.device)
    optimizer = optim.Adam([{'params':model.params_imp,'weight_decay':args.imp_wd, 'lr': args.imp_lr},
                            {'params':model.params_exp,'weight_decay':args.exp_wd, 'lr': args.exp_lr}])
    model.to(device)


##########################################################################################################
    features = S
    true_labels = gt_reshape
    adj = sp.csr_matrix(A_ones)
    args.cluster_num=num_classes
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    print('Laplacian Smoothing...')
    adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
    sm_fea_s = sp.csr_matrix(features).toarray()
    sm_fea_s = torch.FloatTensor(sm_fea_s)
    adj_1st = (adj + sp.eye(adj.shape[0])).toarray()

    model2 = my_model([features.shape[1]] + args.dims,features.shape[0],num_classes)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr)
    model2 = model2.to(device)
    inx = sm_fea_s.to(device)
    target = torch.FloatTensor(adj_1st).to(device)
    #############################################################################################

    for seed in range(10):
        # setup_seed(seed)
        if args.train:
            cnt_wait = 0
            best_loss = 1e9
            best_epoch = 0
            best_acc = 0
            EYE = torch.eye(args.N).to(args.device)
            for epoch in range(1,args.epochs+1):
                model.train()
                model2.train()
                optimizer.zero_grad()
                optimizer2.zero_grad()

                ####################################################################################
                emb = model(knn, adj_norm, norm_factor)
                loss =args.gamma*nn.MSELoss()(torch.mm(emb,emb.t()), EYE)/args.N
                aaa=torch.mm(emb, emb.t())
                loss.backward()
                optimizer.step()
                optimizer2.step()
                ####################################################################################

                ####################################################################################
                z1, z2, q, p = model2(inx, emb.detach(),is_train=True, sigma=args.sigma2)
                kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
                S = z1 @ z2.T
                loss2 = F.mse_loss(S, target)+1000*kl_loss
                loss2.backward()
                ####################################################################################


                if loss <= best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    cnt_wait = 0
                else:
                    cnt_wait += 1
                if cnt_wait == args.patience or math.isnan(loss):
                    print('\nEarly stopping!', end='')
                    break

                ####################################################################################
                model2.eval()
                z1, z2,_,_ = model2(inx, emb.detach(), is_train=False, sigma=args.sigma2)
                hidden_emb = (z1 + z2) / 2
                Y_emb = torch.cat([emb, hidden_emb], dim=-1)
                predict_labels = KMeans(n_clusters=num_classes).fit_predict(Y_emb.cpu().detach().numpy())

                indx = np.where(gt_reshape != 0)
                labels = gt_reshape[indx]

                pixel_y = spixel_to_pixel_labels(predict_labels, Q)
                prediction = pixel_y[indx]
                acc, kappa, nmi, ari, pur, ca = cluster_accuracy(labels, prediction, return_aligned=False)
                print('\nEpoch3：{} acc3: {:.2f}, nmi3: {:.2f}, ari3: {:.2f}'.format(epoch, acc * 100, nmi, ari))

                f = open('./results/' + 'sa' + '_results.txt', 'a+')

                str_results = '\n\n************************************************' \
                              + '\nepoch={}'.format(epoch) \
                              + '\nacc={:.4f}'.format(acc) \
                              + '\nkappa={:.4f}'.format(kappa) \
                              + '\nnmi={:.4f}'.format(nmi) \
                              + '\nari={:.4f}'.format(ari) \
                              + '\npur={:.4f}'.format(pur) \
                              + '\nca=' + str(np.around(ca, 4)) \



                f.write(str_results)
                f.close()