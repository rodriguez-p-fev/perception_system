import argparse

parser = argparse.ArgumentParser(description='Track detection')

parser.add_argument('-path', '--main_path', 
                    type=str, 
                    help='Main path directory',
                    required=True)
parser.add_argument('-mode', '--detection_mode', 
                    type=str, 
                    help='Main path directory',
                    required=False, 
                    default="normal",   
                    choices=["normal","wayside"])

args = parser.parse_args()