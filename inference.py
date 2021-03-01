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
    parser.add_argument('--submit_type',type=str, default=None, help='select sumbission type [csv, package]')
    parser.add_argument('--cv_type', type=str, help='select cv type [SGCV, GTCV, random(for Resnet 1dcnn only)]')
    parser.add_argument('--selection', type=str, help='select params figured out [1, 2] (ResnetLinear, EmbedNN only)')
    parser.add_argument('--pretrained', type=str2bool, help='select to use pretrained weight for inference')

    args = parser.parse_args()
    model_type = args.model_type
    submit_type = args.submit_type
    cv_type = args.cv_type
    selection = args.selection
    pretrained = args.pretrained

    if model_type == "Resnet1dcnn":
        selection = None

    data_utils = DataUtils(entire_seed=1029)
    X, y, cv_idxes = data_utils.prepare_data(cv_type)

    if pretrained == False:

        model = data_utils.train(model_type,X,y,cv_idxes, selection, epochs=1, batch_size=4096, early_stopping=True)
    else:
        model, optimizer, learning_rate, weight_decay = data_utils.set_params(model_type,selection)

    if submit_type == "csv":
        csv_submission = data_utils.csv_inference(model, model_type)
        data_utils.save_csv(csv_submission,f"{model_type}_{cv_type}_{selection}_submission.csv")
    elif submit_type == "package":
        data_utils.package_inference(model, model_type)

if __name__ == '__main__':
    main()