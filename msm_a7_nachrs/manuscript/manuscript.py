import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import matplotlib.collections as clt
import matplotlib.pyplot as plt
import seaborn as sns
import ptitprince as pt
sns.set(style="ticks", context='paper', font_scale=2)
sns.despine()

scatter_kws = {'s': 200, 'linewidths': 3}

line_kws = {'linewidth': 4}

colors = [(0, 0, 0), (0.25, 0.36, 0.46), (0.17, 0.51, 0.73)]
cmap_name = 'open'
cmap_open = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

colors = [(0, 0, 0), (0.40, 0.41, 0.33), (0, 0.52, 0.44)]
cmap_name = 'closed'
cmap_closed = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

colors = [(0, 0, 0), (0.78, 0.56, 0.32), (0.65, 0.38, 0.10)]
cmap_name = 'desensitized'
cmap_de = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

colors = [(1, 1, 1), (0.25, 0.36, 0.46), (0.17, 0.51, 0.73)]
cmap_name = 'open'
cmap_open_light = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

colors = [(1, 1, 1), (0.40, 0.41, 0.33), (0, 0.52, 0.44)]
cmap_name = 'closed'
cmap_closed_light = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

colors = [(1, 1, 1), (0.78, 0.56, 0.32), (0.65, 0.38, 0.10)]
cmap_name = 'desensitized'
cmap_de_light = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

struc_2_state = {
    'EPJPNU': 'open',
    'BGT': 'closed',
    'EPJ': 'desensitized',
}
state_color_dict = {
    'open': '#3F5C76',
    'closed': '#656855',
    'desensitized': '#C78E52',
    'I': '#4A235A',
    'flipped': '#4DB6AC'
    }

system_color_dict = {
    'apo': '#FB3640',
    'cholesterol': '#80BF42'
    }

def set_axis_boarder(ax):
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['bottom'].set_linewidth(4)
    ax.xaxis.set_tick_params(width=4)
    ax.yaxis.set_tick_params(width=4)