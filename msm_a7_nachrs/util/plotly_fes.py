import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import sys

from ENPMDA import MDDataFrame

def generate_tica_csv(md_dataframe,
                      msm_obj,
                      sel_tic1='tica_1',
                      sel_tic2='tica_2',
                      output='tica.csv'):
    plotly_tica_output = msm_obj.transform_feature_trajectories(md_dataframe)
    plotly_df = md_dataframe.dataframe

    plotly_tica_concatenated = np.concatenate(plotly_tica_output[::5])
    plotly_df[sel_tic1] = plotly_tica_concatenated[:, 0]
    plotly_df[sel_tic2] = plotly_tica_concatenated[:, 1]

    plotly_df['msm_weight'] = 0
    plotly_df.iloc[plotly_df[plotly_df.frame >= msm_obj.start * msm_obj.md_dataframe.stride].index, -1] = np.concatenate(msm_obj.msm.trajectory_weights()[::5])
    plotly_df.to_csv(output)


# copy from pyemma plot2d
def _to_free_energy(z, minener_zero=False):
    """Compute free energies from histogram counts.
    Parameters
    ----------
    z : ndarray(T)
        Histogram counts.
    minener_zero : boolean, optional, default=False
        Shifts the energy minimum to zero.
    Returns
    -------
    free_energy : ndarray(T)
        The free energy values in units of kT.
    """
    pi = z / float(z.sum())
    free_energy = np.inf * np.ones(shape=z.shape)
    nonzero = pi.nonzero()
    free_energy[nonzero] = -np.log(pi[nonzero])
    if minener_zero:
        free_energy[nonzero] -= np.min(free_energy[nonzero])
    return free_energy

def export_plotly(tica_csv,
                  output):
    print('tica_csv: ', tica_csv)
    print('output: ', output)

    # load tica data
    try:
        plotly_df = pd.read_csv(tica_csv)
    except:
        print(f'No tica data found. Generate {tica_csv} first.')
        exit()

    struc_state_dic = {
            'BGT': 'CLOSED',
            'EPJ': 'DESENSITIZED',
            '7ekt': 'I (7EKT)',
            'EPJPNU': 'OPEN',
    }

    x = plotly_df['tica_1']
    y = plotly_df['tica_2']
    weights = plotly_df['msm_weight']

    z, xedge, yedge = np.histogram2d(
            x, y, bins=100, weights=weights)

    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])

    z = np.maximum(z, np.min(z[z.nonzero()])).T
    f = _to_free_energy(z, minener_zero=True)

    fig = go.Figure(go.Contour(
        x=x,
        y=y,
        z=f,
        zmax=10,
        zmin=0,
        zmid=3,
        ncontours=20,
        colorscale = 'Earth',
        showscale=False)
    )

    fig.update_traces(contours_coloring="fill", contours_showlabels = True)

    for system, df in plotly_df.groupby('system'):
        pathway = df.pathway.unique()[0]
        pathway_text = ' to '.join([struc_state_dic[path][0] for path in pathway.split('_')])
        seed = df.seed.unique()[0]
        x = df[df.traj_time%10000 == 0]['tica_1'].values
        y = df[df.traj_time%10000 == 0]['tica_2'].values
        t = df[df.traj_time%10000 == 0]['frame'].values

        weights = df[df.traj_time%10000 == 0]['msm_weight'].values
        fig.add_trace(
            go.Scatter(x=x, y=y,
                name=f'SEED_{seed}',
                mode='markers',
                legendgroup=pathway_text,
                legendgrouptitle_text=pathway_text,
                showlegend=True,
                marker=dict(
                color=t,
                colorscale='Purp',
                size=10,
                opacity=1,
                showscale=False)
            )
        )

        if seed == 0 and pathway in ['BGT_EPJPNU',
                                    'EPJ_EPJPNU',
                                    'EPJPNU_BGT']:
            fig.add_annotation(x=x[10], y=y[10],
            text=struc_state_dic[pathway.split('_')[0]],
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
                ),
            align="center",
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#ff7f0e",
            opacity=0.8)


    fig.update_xaxes(title_text="IC 1")
    fig.update_yaxes(title_text="IC 2")
    fig.update_layout(
        autosize=True,
        width=1300,
        height=700,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
        paper_bgcolor="LightSteelBlue",
    )

    fig.update_layout(
    xaxis_range=[-1.5, 1.3],
    yaxis_range=[-2.1, 2.9],
    legend=dict(x=1.1, y=0.95),
    legend_groupclick="toggleitem",
    legend_orientation="h")

    with open(output, 'w') as f:
        f.write(fig.to_html(full_html=True, include_plotlyjs='cdn'))

    print(f'Exported {output}')

def main(args):
    import argparse

    parser = argparse.ArgumentParser(description='Export plotly html from tica csv file')
    parser.add_argument('-tica_csv', type=str, default='plotly_tica.csv')
    parser.add_argument('-output', type=str, default='plotly_fes.html')
    args = parser.parse_args()
    export_plotly(tica_csv=args.tica_csv,
                    output=args.output)

if __name__ == "__main__":
   main(sys.argv[1:])