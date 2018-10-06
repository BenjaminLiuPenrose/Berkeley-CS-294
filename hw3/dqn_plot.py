import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json
import os

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data):

    sns.set(style="darkgrid", font_scale=1.5)
    xs = list(np.linspace(1, len(data), len(data))*20000+50000)
    plt.plot(xs, data)
    plt.legend(loc='best').draggable()
    plt.show()



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('pkldir', nargs='*')
    args = parser.parse_args()

    data = []
    for pkldir in args.pkldir:
        with open(pkldir, 'rb') as fil:
            data += pickle.load(fil)
        plot_data(data)

if __name__ == "__main__":
    main()
