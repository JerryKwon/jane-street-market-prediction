import argparse
import warnings

from tqdm import tqdm

from data_utils import DataUtils

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="execute train.py")
    parser.add_argument('--model_type', type=str, default=None, help='select model type [Resnet1dcnn, ResnetLinear, EmbedNN]')
    parser.add_argument('--cv_type', type=str, help='select cv type [SGCV, GTCV, random(for Resnet 1dcnn only)]')
    parser.add_argument('--selection', type=int, help='select params figured out [1, 2] (ResnetLinear, EmbedNN only)')

    args = parser.parse_args()
    model_type = args.model_type
    cv_type = args.cv_type
    selection = args.selection

    if model_type == "Resnet1dcnn":
        selection = None

    data_utils = DataUtils(entire_seed=1029)
    X,y, cv_idxes = data_utils.prepare_data(cv_type)

    model = data_utils.train(model_type, X, y, cv_type, cv_idxes, selection, epochs=100, batch_size=4096, early_stopping=7)

if __name__ == '__main__':
    main()