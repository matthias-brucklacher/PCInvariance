"""Create final RDM figure after runnning train.py.

Example:
    $ python post_run_analysis/plot_rdm.py --simulation_id <simulation_id> (optional: --mode <mode>)

"""
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

# Get plotting style and simulation name
parser = argparse.ArgumentParser(description ='Create RDM plot')
parser.add_argument('--mode', type = str, nargs = '?', default = 'paper', help = 'paper (default) or poster (larger labels)')
parser.add_argument('--simulation_id', type = str, nargs = '?', help = 'Directory with file rdm.npy')
args = parser.parse_args()

# Load file
rdm = np.load('../results/intermediate/' + args.simulation_id + '/rdm.npy')
 
# Plot RDM 
fig, ax = plt.subplots()
labelsize = 13
if args.mode == 'paper':
    labelsize = 18
size = 18
im = ax.matshow(rdm, vmin=0, vmax = 1)
ax.xaxis.set_ticks_position('bottom')
ax.set_xlabel('Sample #', size=size)
ax.set_ylabel('Sample #', size=size)
ax.tick_params(axis='both', which='major', labelsize=labelsize)
cbar_ax = fig.add_axes([0.73, 0.17, 0.05, 0.7])
if args.mode == 'poster':
    cbar_ax.tick_params(labelsize=labelsize-3)
fig.colorbar(im, cax=cbar_ax).set_label(label='Cosine Dissimilarity', size=size)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(right=0.75)

# Save figure
figures_path = f'../results/figures/{args.simulation_id}'
if not os.path.exists(figures_path):
    os.mkdir(figures_path)
plt.savefig(figures_path + '/rdm.svg')

plt.show()
