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

class TrainModel:


def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="execute train.py")
    parser.add_argument('--model_type', type=str, default=None, help='select model type [Resnet1dcnn, ResnetLinear, EmbedNN]')
    parser.add_argument('--is_valid', type=str2bool, help='select dataset [True=split from train.csv, False=example_test.csv]')
    parser.add_argument('--cv_type', type=str, help='select cv type [SGCV, GTCV, random(for Resnet 1dcnn only)]')
    parser.add_argument('--selection', type=str, help='select params figured out [1, 2] (ResnetLinear, EmbedNN only)')

    args = parser.parse_args()
    model_type = args.model_type
    is_vaild = args.is_valid
    cv_type = args.cv_type
    selection = args.selection

    if model_type == "Resnet1dcnn":
        selection = None

    data_utils = DataUtils(entire_seed=1029)
    X,y, cv_idxes = data_utils.prepare_data(cv_type)

    data_utils.train(model_type,X,y,cv_idxes, selection)

    if is_vaild:
        target = data_dict["val"]
    else:
        target = data_dict["test"]

    if model_type == "icbf":
        icbf_model = ICBF_OCC(train_df=data_dict["train"], test_df=target)
        icbf_rcomm_result = icbf_model.execute_recommendation()
        data_loader.write_json(icbf_rcomm_result, 'icbf_rcomm_results.json')

    if model_type == "hybrid":
        hybrid_model = HYBRID_CBF_ICBF(train_df=data_dict["train"], test_df=target)
        hybrid_rcomm_result = hybrid_model.execute_recommendation()
        data_loader.write_json(hybrid_rcomm_result, 'hybrid_rcomm_results.json')


if __name__ == '__main__':
    main()