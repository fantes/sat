import argparse
from models.graphcnnsat import *
from utils.utils import *
from data.graphDataset import *
import glob




def train(model,dataset,test_split, max_epoch, batch_size=1, print_inter = 10, test_inter = 1, num_workers=0, graph_classif = False, device=torch.device("cuda:0")):
    model.to(device)
    model.train()

    train_dl, test_dl, weights = dataset.getDataLoaders(batch_size, test_split, model.maxclause, model.maxvar, model.varvar, graph_classif, num_workers)

    optimizer = torch.optim.AdamW(model.parameters(),amsgrad=True)
    # class 0 is much more present then others
    class_weights = torch.tensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    running_loss = 0.0

    for epoch in range(0, max_epoch):

        for i_batch, sample_batched in enumerate(train_dl):
            batch_size, biggraph, clause_feat, var_feat, graph_pooler, labels = sample_batched
            target = convert_labels(labels, dataset.neg_as_link, model.maxvar, device)
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
            loss = criterion(torch.transpose(res,1,2), target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i_batch % print_inter == print_inter-1:    # 2000 / 1999 print every 2000 mini-batches
                print('[%d, %5d] loss: %.10f' %  (epoch + 1, i_batch * batch_size + 1, running_loss / print_inter))
                running_loss = 0.0

        # test after every epoch
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            precision = []
            recall = []
            f1 = []
            for sample_batched in train_dl:
                batch_size, biggraph, clause_feat, var_feat, graph_pooler, labels = sample_batched
                target = convert_labels(labels, dataset.neg_as_link, model.maxvar, device)
                biggraph = biggraph.to(device)
                clause_feat = clause_feat.to(device)
                var_feat = var_feat.to(device)
                out = model.forward(batch_size, biggraph, clause_feat, var_feat, None)
                res = torch.argmax(out,dim=2)
                for bi in range(len(labels)):
                    preds = []
                    for i,r in enumerate(res[bi]):
                        if r == 1:
                            preds.append(-(i+1))
                        elif r == 2:
                            preds.append(i+1)

                    tp = 0
                    for p in preds:
                        if p in labels[bi]:
                            tp += 1
                    if len(preds) == 0:
                        local_precision = 0
                    else:
                        local_precision = tp/len(preds)
                    precision.append(local_precision)
                    local_recall = tp / len(labels[bi])
                    recall.append(local_recall)
                    if (local_precision + local_recall) == 0:
                        f1.append(0.0)
                    else:
                        f1.append(2 * local_precision * local_recall / (local_precision+local_recall))
            print("precision: " + str(sum(precision)/len(precision)))
            print("recall: " + str(sum(recall)/len(recall)))
            print("f1: " + str(sum(f1)/len(f1)))
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
    parser.add_argument('--output_dim', type=int, default=3,
                        help='output dimension (default: 3 , for true/false/not in clause)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout for graph embedding (default: 0.5)')
    parser.add_argument('--random', type=int, default=1,
                        help='number of random features (default: 1)')
    parser.add_argument('--maxclause', type=int, default=6000,
                        help='max number of clauses (default: 20M)')
    parser.add_argument('--maxvar', type=int, default=2000,
                        help='max number of vars (default: 5M)')
    parser.add_argument('--graph_type', type=str, default="clause.var",
                        help='graph_type in clause.var|var.var')
    parser.add_argument('--var_classif', type=bool, default=True,
                        help='goal is var classification (default: True)')
    parser.add_argument('--clause_classif', type=bool, default=False,
                        help='goal is clause classification (default: False) can be combined with var classif , both false means whole graph classif')
    parser.add_argument('--pgso', action='store_true',  help='use pgso')
    parser.add_argument('--mpgso', action='store_true',  help='use mpgso')
    parser.add_argument('--graph_norm', type=bool, default=True,
                        help='use graph normalization')
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

    args = parser.parse_args()

    model = GraphCNNSAT(args.num_layers, args.num_mlp_layers,  args.hidden_dim, args.output_dim, args.final_dropout, args.random, args.maxclause, args.maxvar, args.graph_type, args.var_classif, args.clause_classif, False, args.pgso, args.mpgso, args.graph_norm, args.neighbor_pooling_type, args.graph_pooling_type, args.lfa)
    model.to(torch.device("cuda:0"))

    tds = GraphDataset(glob.glob(args.graphfile), cachesize=0, path_prefix=args.datasetpath)

    train(model, tds, args.test_split,  args.epoch, args.batch_size)


if __name__ == '__main__':
    main()
