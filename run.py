import argparse
from data import dataset
from model import model
from model import trainer
from utils import utils
import questionary
import torch
import os


def main(train_data_path, data_paths, version, config_args, train_args, func, save_dir, pretrain_state=None):

    if pretrain_state:
        state_dict = pretrain_state['state_dict']
    else:
        state_dict = None

    # get device
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    print("DEVICE: {}".format(device))

    # build datasets
    print('\nProcessing dataset...')

    train_dataset = dataset.Directory(train_data_path,
                                      data_paths,
                                      version,
                                      config_args)()
    # load model
    mconf = model.GPTConfig(
        vocab_size=train_dataset.vocab.vocab_size,
        args_dict=config_args
    )

    # build model
    gpt_model = model.GPT(mconf)
    gpt_model = gpt_model.to(device)

    train_config = trainer.TrainerConfig(func=func,
                                         state_dict=state_dict,
                                         args_dict=train_args)

    model_trainer = trainer.Trainer(gpt_model,
                                    train_dataset,
                                    save_dir,
                                    config=train_config)
    model_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('function', type=str,
                        help='Pretrain or finetune model.',
                        choices=["pretrain", "finetune"])
    parser.add_argument('--version', type=int, default=None,
                        help='Finetune version.')
    parser.add_argument('--data_dir', type=str,
                        help='Dataset directory to use.')
    parser.add_argument('--save_dir', type=str,
                        help='Directory to save checkpoints.')

    # definitely use pretrain params when finetuning
    parser.add_argument('--pretrain_params', type=str,
                        help='Path to model params (use for finetune).')
    parser.add_argument('--args_path', type=str,
                        help='Path to JSON training args.')
    parser.add_argument('--block_size', type=int,
                        help='Super config arg.')
    parser.add_argument('--n_layer', type=int,
                        help='Super config arg.')
    parser.add_argument('--n_head', type=int,
                        help='Super config arg.')
    parser.add_argument('--n_embed', type=int,
                        help='Super config arg.')
    parser.add_argument('--max_epochs', type=int,
                        help='Super train arg.')
    parser.add_argument('--batch_size', type=int,
                        help='Super train arg.')
    parser.add_argument('--learning_rate', type=float,
                        help='Super train arg.')
    parser.add_argument('--num_workers', type=int,
                        help='Super train arg.')

    # WARNING: individual args superceded ARGS file
    args = parser.parse_args()

    # Double check args
    data_dir = args.data_dir
    save_dir = args.save_dir
    func = args.function
    version = args.version

    possible_versions = list(dataset.finetune_versions.keys())

    if not version:
        version = 0

    elif not version and func == 'finetune':
        print('WARNING: FINETUNING WITHOUT A VERSION')
        print('SETTING TO DEFAULT FINETUNE VERSION 0')
        version = 0

    if not data_dir:
        def_data = 'data/datasets-cleaned'

        answer = questionary.confirm(f'Use default data directory--{def_data}?').ask()
        if answer:
            data_dir = 'data/datasets-cleaned'
            assert os.path.isdir(data_dir), 'DATA FILE NOT FOUND'
        else:
            raise FileExistsError('Must provide a dataset for training!')

    data_dir_list = [os.path.join(data_dir, x) for x in os.listdir(data_dir) if x[0] != '.']

    if not save_dir:
        save_dir = os.path.join('ckpts', func + '_default')

        answer = questionary.confirm(f'Use save directory at {save_dir}?').ask()
        if not answer:
            save_dir = questionary.text('Enter checkpoint save directory: ').ask()

    assert not os.path.isfile(save_dir), 'Directory cannot be a file!'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if func == 'pretrain' and args.pretrain_params:
        assert questionary.confirm('Pretrain is provided with pretrain params. Continue?').ask()
    if func == 'finetune' and not args.pretrain_params:
        if version != 3:
            raise ValueError('Cannot finteune without a pretrained model!')

    # Get args if provided for finetune
    if func == 'finetune' and args.pretrain_params:
        pretrain_dict = torch.load(args.pretrain_params)
        pretrain_model_config = pretrain_dict['model_config']
        pretrain_train_config = pretrain_dict['train_config']
        pretrain_args = pretrain_model_config.__dict__.update(pretrain_train_config.__dict__)
    else:
        pretrain_args = None
        pretrain_dict = None

    # Check config args
    meta_args = ['data_dir', 'save_dir', 'function', 'pretrain_params']
    super_config_train_args = {key: val for key, val in vars(args).items() if key not in meta_args}     

    default_config_args = utils.default_config_args
    default_train_args = utils.default_train_args

    # No provided args
    if func == 'pretrain':
        if len(set(super_config_train_args.values())) == 1 and not set(super_config_train_args.values()).pop() and not args.args_path:
            print('NO ARGS PROVIDED. USING DEFAULT ARGS\n')
            print("Config Args:", default_config_args)
            print("Train Args:", default_train_args)

    # Mixed args
    if pretrain_args and (len(set(super_config_train_args.values())) > 1 or args.args_path):
        print('WARNING: DO NOT CHANGE MODEL CONFIGURATION FOR FINETUNING')

    # get separate updated config and train args
    arguments = utils.TrainArgs(args.args_path, super_config_train_args, pretrain_args=pretrain_args)
    config_args, train_args = arguments()

    # map version to train data:
    v2d = {0: os.path.join(data_dir, 'kaggle_cleaned.txt')}
    train_data_path = v2d[version]

    assert os.path.isfile(train_data_path)

    main(train_data_path, data_dir_list, version, config_args, train_args, func, save_dir, pretrain_state=pretrain_dict)
    4943104