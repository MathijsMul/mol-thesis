import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.font_manager import FontProperties
import os
import seaborn as sns

DET_SUBJ_LIST = ['some', 'all', 'not_some', 'not_all']
#MAX_GRU_SCORE = 96.89
gru_best = 97
srn_best = 83
#MODEL = 'gru'

#dir = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/logs/binary/2dets_4negs/hierarchic_gen/' + MODEL + '/binary_2dets_4negs_' + MODEL + '_'
dir = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/logs/binary/2dets_4negs/hierarchic_gen/gru/segment_bulk_2det_4negs/binary_2dets_4negs_gru_frombulk_'
#dir = '/Users/mathijs/Documents/Studie/MoL/thesis/mol_thesis/logs/binary/2dets_4negs/hierarchic_gen/srn/binary_2dets_4negs_srn_'

all_results = []

for item1 in DET_SUBJ_LIST[::-1]:
    results = []
    for item2 in DET_SUBJ_LIST:
        filename = dir + item1 + item2 + '.txt'
        with open(filename, 'r') as f:
            for idx, line in enumerate(f):
                if idx == 79:
                    acc = float(line.split()[1])
                    results.append(acc)

    all_results.extend([results])

result = np.array(all_results)

print(result)
print(np.average(result))
# print(np.average(result - srn_best))

exit()

color_palette = sns.color_palette('Blues', 255, desat=.8)
ax = sns.heatmap(result, annot=True, xticklabels=DET_SUBJ_LIST, yticklabels=DET_SUBJ_LIST[::-1], cmap=color_palette)
plt.xlabel('s2 det_subj')
plt.ylabel('s1 det_subj')
plt.savefig('hierarchic_vis_gru_frombulk')

if False:
    # generate a ridiculous 3d plot

    fig=plt.figure(figsize=(5, 5), dpi=150)
    ax1=fig.add_subplot(111, projection='3d')

    # os.environ['PATH'] = '/Library/TeX/texbin'
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    #det_subj_list_tex = [r'none', r'\textit{some}', r'\textit{all}', r'\textit{not some}', r'\textit{not all}']
    xlabels = np.array(DET_SUBJ_LIST)
    xpos = np.arange(xlabels.shape[0])
    ylabels = np.array(DET_SUBJ_LIST)
    ypos = np.arange(ylabels.shape[0])

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos=result
    zpos = zpos.ravel()

    dx=0.5
    dy=0.5
    dz=zpos

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }

    ax1.w_xaxis.set_ticks(xpos - 0.5*dx)
    ax1.w_xaxis.set_ticklabels(xlabels)
    ax1.set_xlabel('s2')

    ax1.w_yaxis.set_ticks(ypos + 1.5*dy)
    ax1.w_yaxis.set_ticklabels(ylabels)
    ax1.set_ylabel('s1')

    ax1.w_zaxis.set_ticks([20,40,60,80])

    values = np.linspace(0.2, 1., xposM.ravel().shape[0])
    colors = cm.rainbow(values)
    ax1.bar3d(xposM.ravel(), yposM.ravel(), dz*0, dx, dy, dz, color=colors)

    plt.show()