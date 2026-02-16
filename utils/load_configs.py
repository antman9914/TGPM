import argparse
import sys
import torch


def get_link_prediction_args():

    # arguments
    parser = argparse.ArgumentParser('Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', choices=['icews','enron','googlemap'], default='icews')
    parser.add_argument('--eval_dataset', type=int, default=0, help="Dataset for evaluation")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--run_name', type=str, default='TGPM', help='name of the run, which will determine saving path.')
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=32, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk', type=int, default=8, help='number of sampled random walks.')
    parser.add_argument('--num_heads', type=int, default=12, help='number of heads used in attention layer')
    parser.add_argument('--encoder_layers', type=int, default=2, help='number of encoder layers')
    parser.add_argument('--decoder_layers', type=int, default=1, help="number of decoder layers")
    parser.add_argument('--hidden_dim', type=int, default=768, help='dimension of hidden layer')
    parser.add_argument('--walk_length', type=int, default=6, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--pred_lr', type=float, default=0.0005, help="learning rate for prediction head")
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=5, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--pre_train', action='store_true', default=False, help="whether perform pre-training")
    parser.add_argument('--ft', action="store_true", default=False, help="whether fine-tune the pre-trained model.")
    parser.add_argument('--inference_only', action="store_true", default=False, help="whether skip pre-train and do inference")
    parser.add_argument('--aug_data', action="store_true", default=False, help="Whether utilize data augmentation during pre-training")
    parser.add_argument('--no_ntp', action="store_true", default=False, help="Whether use NTP pre-training")
    parser.add_argument('--no_lt', action="store_true", default=False, help="Whether use long-term pre-training")
    parser.add_argument('--no_st', action="store_true", default=False, help="Whether use short-term pre-training")
    parser.add_argument('--random_mask', action="store_true", default=False, help="Whether use random masking.")
    parser.add_argument('--causal_path', action="store_true", default=False, help="Whether use causal path.")
    parser.add_argument('--block_size', type=int, default=6, help="Size of masked block.")
    parser.add_argument('--visible_ratio', type=float, default=0.25, help="Ratio of visible interaction patches.")

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    return args
