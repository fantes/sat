import argparse
from models.graphcnnsat import GraphCNNSAT
from utils.utils import *





def train(model,dataset,batch_size, num_workers, graph_classif = False):
    model.train()

    dl = dataset.getDataLoader(batch_size, model.maxclause, model.maxvar, model.half_compute, graph_classif, num_workers)

    for i_batch, sample_batched in enumerate(dataloader):
        batch_size, biggraph, clause_feat, var_feat, graph_pooler, labels = sample_batched
        biggraph.to(model.device)
        clause_feat.to(model.device)
        var_feat.to(model.device)
        if graph_pooler is not None:
            graph_pooler.to(model.device)
        res = model.forward(batch_sizen biggraph, clause_feat, var_feat, graph_pooler)


def main():
    parser = argparse.ArgumentParser(description='sat trainer')
    parser.add_argument('--num_layers', type=int, default=10,
                                help='number of layers (neighborhood depth)  (default: 10)')
    parser.add_argument('--num_mlp_layers', type=int, default=2,
                        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument('--hidden_dim', type=int, default=8,
                        help='number of hidden units (default: 8)')
    parser.add_argument('--output_dim', type=int, default=2,
                        help='output dimension (default: 2 , for true/false)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout for graph embedding (default: 0.5)')
    parser.add_argument('--random', type=int, default=1,
                        help='number of random features (default: 1)')
    parser.add_argument('--maxclause', type=int, default=20000000,
                        help='max number of clauses (default: 20M)')
    parser.add_argument('--maxvar', type=int, default=5000000,
                        help='max number of vars (default: 5M)')
    parser.add_argument('--half', type=bool, default=True,
                        help='use half computation (smaller matrices, 2x times more matrices product) (default: True)')
    parser.add_argument('--var_classif', type=bool, default=True,
                        help='goal is var classification (default: True)')
    parser.add_argument('--clause_classif', type=bool, default=False,
                        help='goal is clause classification (default: False) can be combined with var classif , both false means whole graph classif')
    parser.add_argument('--graph_pooling_type', type=str, default="sum", choices=["sum", "average"],
                        help='Pooling for over nodes in a graph: sum or average')
    parser.add_argument('--neighbor_pooling_type', type=str, default="sum", choices=["sum", "average"],                        help='Pooling for over neighboring nodes: sum, average')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU id (default:0)')
    parser.add_argument('--graphfile', type=str, help='graph file as obtained by preprocess.py')
    parser.add_argument('--datasetpath', type=str, help='prefix of partial graphs')

    args = parser.parse_args()

    model = GraphCNNSAT(args.num_layers, args.num_mlp_layers,  args.hidden_dim, args.output_dim, args.final_dropout, args.random, args.maxclause, args.maxvar, args.half, args.var_classif, args.clause_classification, args.neighbor_pooling_type, args.graph_pooling_type, torch.device("cuda:"+str(args.device)))

    tds = GraphDataset(args.graphfile, cachesize=0, path_prefix=args.datasetpath)

    train(model, tds)


if __name__ == '__main__':
    main()
