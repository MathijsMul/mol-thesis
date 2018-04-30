import matplotlib.pyplot as plt
import os
an = [0.1, 0.75]
of = [0.2, 0.72]

illustration = [0.3, 0.45]
distributional = [0.55, 0.55]
semantics = [0.57, 0.6]

points = [an, illustration, of, distributional, semantics]

os.environ['PATH'] = '/Library/TeX/texbin'

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

for point in points:
    plt.scatter(point[0], point[1],c='black')

plt.annotate('an', [0.12,0.75],fontsize=15)
plt.annotate('of', [0.22,0.72],fontsize=15)

plt.annotate('illustration', [0.32,0.45],fontsize=15)
plt.annotate('distributional', [0.35,0.55],fontsize=15)
plt.annotate('semantics', [0.42,0.6],fontsize=15)

plt.xlim(0,0.75)
plt.ylim(0.4,0.8)
plt.savefig('distributional-meaning',dpi=500)