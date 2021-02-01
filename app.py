import os
import xml
import pandas as pd
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from sklearn.neighbors import NearestNeighbors
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import plotly.express as px

def calc_cosine_sim(df):
    """Function to calculate cosine similarity between all pairwise sets of points in an array"""
    dot_prod = np.dot(df, df.T)
    dot_prod /= np.dot(np.linalg.norm(pca_df, axis=1).reshape(-1,1), np.linalg.norm(pca_df, axis=1).reshape(1,-1))
    return 1-dot_prod

def calc_euclidean_sim(df):
    """Function to calculate euclidean distance between all pairwise sets of points in an array"""
    results = -2 * np.dot(df, df.T)
    A_squared = np.sum(df**2, axis=1).reshape(-1,1)
    results += A_squared
    B_squared = np.sum(df**2, axis=1)
    results += B_squared

    return np.sqrt(results)


def find_n_most_similar_indices(similarity_df, player_idx, n):
    """Returns the indices of the n most similar players to the given player"""
    return np.argsort(similarity_df[player_idx])[:n]

def match_player_names(match_indices, player_df):
    """Extract the rows of data corresponding to the indices"""
    matches = player_df.iloc[match_indices]
    return matches


def top_n_players(df, player_df, n, player_idx, distance_metric='cosine'):
    """Return the names and stats of  the n most similar players"""
    if distance_metric == 'cosine':
        sim = calc_cosine_sim(df)
    elif distance_metric == 'euclidean':
        sim = calc_euclidean_sim(df)

    n_indices = find_n_most_similar_indices(sim, player_idx, n)

    similar_players = match_player_names(n_indices, player_df)

    return similar_players, n_indices

# Read data from csv file (data already processed with SQL, panda joins etc 
# and saved to csv for quicker processing
analysis_df = pd.read_csv("opta.csv", index_col=0)

# Distinct player names
distinct_player_names = (analysis_df['PLFORN'] + " " + analysis_df['PLSURN'] + " - " + analysis_df.index.astype(str)).values.tolist()


# Scale data to eliminate differences in magnitude of variables
scaled_df = minmax_scale(analysis_df.iloc[:,:-2])

pca = PCA()
pca_df = pca.fit_transform(scaled_df)

##external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)##, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    children=[
        html.H1("Player Stats Similarity"),
        dcc.Dropdown(
            id='player-dropdown',
            options=[
                {'label': player, 'value': player} for player in distinct_player_names
            ],
        ),
        html.Div(id="n-most-similar",
                 style = {'width': '100%',
                          'display':'flex',
                          'align-items': 'center',
                          'justify-content':'center'}
        ),
        html.Div(id="stats-table"),
    ]
)

@app.callback([
    Output("n-most-similar", "children"),
    Output("stats-table", "children")],
    Input('player-dropdown', 'value')
)


def player_similarity_list(player):
    player_name, PLID = player.split(" - ")

    player_idx = np.argmax(analysis_df.index == int(PLID))
    top_n, n_indices = top_n_players(pca_df, analysis_df, 20, player_idx)
    top_n_names = (top_n['PLFORN'] + " " + top_n['PLSURN']).tolist()

    df = analysis_df.iloc[n_indices]
    df['Name'] = df['PLFORN'] + " " + df['PLSURN']
    df.set_index("Name", inplace=True)
    df_transposed = df.iloc[:,:-2].T.reset_index()

    datatable = dash_table.DataTable(
        id='table',
        columns=[{'name':i, 'id':i} for i in df_transposed.columns],
        data = df_transposed.round(2).to_dict('records'),
        style_cell_conditional=[
        {
            'if': {'column_id': c},
            'textAlign': 'left'
        } for c in ['Date', 'Region']
        ],
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(248, 248, 248)'
            }
        ],
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )


    return html.Ul([html.Li(player) for player in top_n_names]), datatable

if __name__=='__main__':
    app.run_server(debug=True)













