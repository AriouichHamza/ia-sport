import argparse

parser = argparse.ArgumentParser(description='Parsing arguments...')

parser.add_argument('--a',type=int,required=True)
parser.add_argument('--b', type=bool,required=True)
parser.add_argument('--c', type=bool,required=True)


args = parser.parse_args()

print("Execution's arguments: ", args)