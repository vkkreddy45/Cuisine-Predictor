# -*- coding: utf-8 -*-
# Example main.py
import argparse
import prediction

def main(args):
    # Getting the input data
    if args.N and args.ingredient:
        print(args.N, args.ingredient)
        prediction.start(args.ingredient,args.N)
    
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--N', action='store', help="Top N Values", required=True)
    parser.add_argument('--ingredient', action='append', help="Inputing Ingrdients", required=True )
    args=parser.parse_args()

    main(args)
