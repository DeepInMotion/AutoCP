import sys
import logging
import platform
import argparse
from omegaconf import OmegaConf


def main():
    # load args from user
    parser = argparse.ArgumentParser(description='AutoCP algorithm')
    parser.add_argument('--mode', '-m', choices=['nas', 'build'], default='nas',
                        help='Select the operating mode')
    parser.add_argument('--config', '-c', default='./config/config_cp.yaml', type=str,
                        help='Choose the config file')
    parser.add_argument('--debug', '-d', action='store_true', help='Debug mode on/off')
    parser.add_argument('--node', '-n', default='./config/config_node.yaml', type=str,
                        help='Choose the node config file')
    parser = parser.parse_args()

    # Overall config
    args = OmegaConf.load(parser.config)

    # get mode and debug flag from parser
    args.mode = parser.mode
    args.debug = parser.debug

    # Load Node configuration and change data paths accordingly
    node = platform.node()
    if node.startswith("idun"):
        node = "idun"
    config_node = OmegaConf.load(parser.node)

    if node in config_node or any(node.startswith(n) for n in config_node):
        # Update paths based on working node else leave to user definition
        args.dataset_args.root_folder = config_node.get(node, {}).get('dataset_args').get('root_folder')
        args.work_dir = config_node.get(node)['work_dir']

    if args.cont_training:
        # cont. training
        cont_dir = args.cont_dir
        assert cont_dir is not None
        args = OmegaConf.load('{}/config.yaml'.format(cont_dir))
        args.cont_training = True
        args.cont_dir = cont_dir

    # Choice of mode
    if parser.mode == "nas":
        from src.autocp import AutoCP
        autocp_nas = AutoCP(args)
        autocp_nas.train_controller()
    elif parser.mode == "build":
        from src.builder import Builder
        builder = Builder(args)
        builder.retrain()
    elif parser.mode == "proxy":
        raise NotImplemented
    else:
        logging.error("Mode: [{}] not known!".format(parser.mode))
        sys.exit()


if __name__ == '__main__':
    main()
