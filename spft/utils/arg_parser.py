import argparse


def parse_args(args_str=None):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--act_channel', type=str, 
                       help='Path to activation channel configuration file')

    parser.add_argument('--enable_static', type=bool, default=False,
                       help='Enable static channel selection for layers')
    
    args = parser.parse_args(args_str)
    
    return args