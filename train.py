import argparse
from models.graphcnnsat import *
from utils.utils import *
from data.graphDataset import *
import glob
from utils.plotter import VisdomLinePlotter
from utils.dice import *


def train(model,dataset,test_split, max_epoch, lr = 0.01, batch_size=1, dice = False, print_inter = 10, test_inter = 1, num_workers=0, graph_classif = False, device=torch.device("cuda:0")):
    model.to(device)
    model.train()

    torch.set_printoptions(profile="full")

    train_dl, test_dl, weights = dataset.getDataLoaders(batch_size, test_split, model.maxclause, model.maxvar, model.varvar, graph_classif, num_workers)

    plotter = VisdomLinePlotter(env_name="sat")

    #optimizer = torch.optim.AdamW(model.parameters(),amsgrad=True)
    print("using lr: " + str(lr))
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    # class 0 is much more present then others
    class_weights = torch.tensor(weights).to(device)
    #criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    if dice:
        criterion = DiceLoss1D()
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    pi = int(print_inter/batch_size)

    running_loss = 0.0

    for epoch in range(0, max_epoch):

        model.train()
        for i_batch, sample_batched in enumerate(train_dl):
            batch_size, biggraph, clause_feat, var_feat, graph_pooler, labels, nvars = sample_batched
            # print("train labels: " +str(labels))
            target = convert_labels(labels, dataset.neg_as_link, dataset.maxvar, device)
            # print("train target: " +str(target))

            biggraph = biggraph.to(device)
            clause_feat = clause_feat.to(device)
            var_feat = var_feat.to(device)

            optimizer.zero_grad()

            if graph_pooler is not None:
                graph_pooler.to(device)
            res = model.forward(batch_size, biggraph, clause_feat, var_feat, None)
            # res is batchsize x num_var|clauses x 3
            # target should be batchsize x num_var  (each cell contains target class)
            #print(res)
            #print("train res: " + str(torch.transpose(res,1,2)))
            #pred= torch.argmax(res,dim=2)
            #print("train pred: " + str(pred))

            loss = criterion(torch.transpose(res,1,2), target)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            #if i_batch % pi == pi-1:    # 2000 / 1999 print every 2000 mini-batches
                # print('[%d, %5d] loss: %.10f' %  (epoch + 1, (i_batch  + 1)*batch_size, running_loss / print_inter))
        if dice:
            plotter.plot('loss', 'train', 'Loss_Dice', epoch+1, running_loss/print_inter)
        else:
            plotter.plot('loss', 'train', 'Loss', epoch+1, running_loss/print_inter)

        running_loss = 0.0

        with torch.no_grad():
            precision = []
            precision_nophase = []
            recall = []
            recall_nophase = []
            f1 = []
            f1_nophase = []
            for sample_batched in train_dl:
                batch_size, biggraph, clause_feat, var_feat, graph_pooler, labels, nvars = sample_batched
                target = convert_labels(labels, dataset.neg_as_link, dataset.maxvar, device)
                biggraph = biggraph.to(device)
                clause_feat = clause_feat.to(device)
                var_feat = var_feat.to(device)
                out = model.forward(batch_size, biggraph, clause_feat, var_feat, None)
                res = torch.argmax(out,dim=2)

                for bi in range(len(labels)):
                    # print("test labels: " + str(labels[bi]))
                    # print("test target: " + str(target[bi]))
                    # print("test res: " + str(res[bi]))
                    preds = []
                    if dataset.neg_as_link:
                        for i,r in enumerate(res[bi]):
                            if r.item() == 1: # neg case
                                preds.append(-(i+1))
                            elif r.item() == 2:
                                preds.append(i+1)
                    else:
                        for i,r in enumerate(res[bi]):
                            if r.item() == 1:
                                if i >= dataset.maxvar: # neg case
                                    preds.append(-(i-dataset.maxvar+1))
                                else:
                                    preds.append(i+1)
                    # print("preds: " + str(preds))
                    tp = 0
                    tp_nophase = 0
                    for p in labels[bi]:
                        # if p in preds or -p in preds:
                        #     tp_nophase += 1
                        if p in preds:
                            tp += 1
                    if len(preds) == 0:
                        local_precision = 0
                        # local_precision_nophase= 0
                    else:
                        local_precision = tp/len(preds)
                        # local_precision_nophase = tp_nophase/len(preds)

                    precision.append(local_precision)
                    # precision_nophase.append(local_precision_nophase)
                    local_recall = tp / len(labels[bi])
                    # local_recall_nophase = tp_nophase / len(labels[bi])
                    recall.append(local_recall)
                    # recall_nophase.append(local_recall_nophase)

                    if (local_precision + local_recall) == 0:
                        f1.append(0.0)
                    else:
                        f1.append(2 * local_precision * local_recall / (local_precision+local_recall))

                    # if (local_precision_nophase + local_recall_nophase) == 0:
                    #     f1_nophase.append(0.0)
                    # else:
                    #     f1_nophase.append(2 * local_precision_nophase * local_recall_nophase / (local_precision_nophase+local_recall_nophase))


            if dice:
                plotter.plot('precision', 'train', 'Precision_dice', epoch+1, sum(precision)/len(precision))
                plotter.plot('recall', 'train_dice', 'Recall_dice', epoch+1, sum(recall)/len(recall))
                plotter.plot('f1', 'train_dice', 'F1_dice', epoch+1, sum(f1)/len(f1))
            else:

                plotter.plot('precision', 'train', 'Precision', epoch+1, sum(precision)/len(precision))
                plotter.plot('recall', 'train', 'Recall', epoch+1, sum(recall)/len(recall))
                plotter.plot('f1', 'train', 'F1', epoch+1, sum(f1)/len(f1))

            # print("precision: " + str(sum(precision)/len(precision)))
            # print("recall: " + str(sum(recall)/len(recall)))
            # print("f1: " + str(sum(f1)/len(f1)))
            # print("precision_nophase: " + str(sum(precision_nophase)/len(precision_nophase)))
            # print("recall_nophase: " + str(sum(recall_nophase)/len(recall_nophase)))
            # print("f1_nophase: " + str(sum(f1_nophase)/len(f1_nophase)))
                    # print("target: " + str(labels[bi]))
                    # print("pred: " + str(preds))



def main():
    parser = argparse.ArgumentParser(description='sat trainer')
    parser.add_argument('--num_layers', type=int, default=10,
                                help='number of layers (neighborhood depth)  (default: 10)')
    parser.add_argument('--num_mlp_layers', type=int, default=3,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='number of hidden units (default: 8)')
    parser.add_argument('--output_dim', type=int, default=2,
                        help='output dimension (default: 3 , for true/false/not in clause)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout for graph embedding (default: 0.5)')
    parser.add_argument('--random', type=int, default=1,
                        help='number of random features (default: 1)')
    parser.add_argument('--maxclause', type=int, default=8000,
                        help='max number of clauses (default: 20M)')
    parser.add_argument('--maxvar', type=int, default=1500,
                        help='max number of vars (default: 5M)')
    parser.add_argument('--graph_type', type=str, default="clause.var",
                        help='graph_type in clause.var|var.var')
    parser.add_argument('--var_classif', type=bool, default=True,
                        help='goal is var classification (default: True)')
    parser.add_argument('--clause_classif', type=bool, default=False,
                        help='goal is clause classification (default: False) can be combined with var classif , both false means whole graph classif')
    parser.add_argument('--pgso', action='store_true',  help='use pgso')
    parser.add_argument('--mpgso', action='store_true',  help='use mpgso')
    parser.add_argument('--graph_norm', action='store_true',   help='use graph normalization')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"],                        help='Pooling for over neighboring nodes: sum, average')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--lfa', action = 'store_true', help='weither to use lfa')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU id (default:0)')
    parser.add_argument('--graphfile', type=str, help='graph file as obtained by preprocess.py')
    parser.add_argument('--datasetpath', type=str, help='prefix of partial graphs')
    parser.add_argument('--epoch', type=int, default = 10, help='number of epoch')
    parser.add_argument('--test_split', default = 0.1, help='test split')
    parser.add_argument('--batch_size', type = int, default = 1, help='batch_size')
    parser.add_argument('--permute_vars', action='store_true', help='do random permutation of vars')
    parser.add_argument('--lr', type=float, default = 0.01,  help='lr')
    parser.add_argument('--dice', action='store_true',  help='use dice loss')
    parser.add_argument('--normalize', action='store_true',  help='normalize arities')

    args = parser.parse_args()

    tds = GraphDataset(glob.glob(args.graphfile), args.maxvar, permute_vars = args.permute_vars, permute_clauses= False, neg_clauses = True, self_supervised = False, normalize = args.normalize, cachesize=0, path_prefix=args.datasetpath)
    if tds.neg_as_link:
        mmv = args.maxvar
        mc = ars.maxclause
    else:
        mmv = 2*args.maxvar
        mc = args.maxclause + args.maxvar

    model = GraphCNNSAT(args.num_layers, args.num_mlp_layers,  args.hidden_dim, args.output_dim, args.final_dropout, args.random, mc, mmv, args.graph_type, args.var_classif, args.clause_classif, False, args.pgso, args.mpgso, args.graph_norm, args.neighbor_pooling_type, args.graph_pooling_type, args.lfa)
    model.to(torch.device("cuda:0"))


    train(model, tds, args.test_split,  args.epoch, args.lr, args.batch_size, args.dice)


if __name__ == '__main__':
    main()
