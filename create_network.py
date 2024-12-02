
import os

os.environ["NETWORKX_AUTOMATIC_BACKENDS"] = "cugraph"

import warnings

warnings.filterwarnings("ignore")

import torchvision

torchvision.disable_beta_transforms_warning()

from typing import List, Tuple

import networkx as nx
import nx_cugraph as cugraph
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

tqdm.pandas()

def clean_text(text_series: pd.Series) -> pd.Series:
    """
    Cleans text data by removing URLs and HTML entities.
    """
    # Remove URLs
    text_series = text_series.str.replace(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', regex=True)

    # Remove HTML
    text_series = text_series.str.replace('&gt;', '')

    return text_series

def add_datetime_columns(df: pd.DataFrame, time_column: str = 'created_utc') -> pd.DataFrame:
    """
    Adds 'Y' (year) and 'YM' (year-month) columns to the DataFrame based on a timestamp column.
    """
    # Convert timestamp to datetime and extract the year
    df['Y'] = pd.to_datetime(df[time_column], unit='s').dt.year

    # Extract year-month in 'YYYY-MM' format
    df['YM'] = pd.to_datetime(df[time_column], unit='s').dt.strftime('%Y-%m')

    return df

def process_subreddit(data_root: str, subreddit_names: List[str], min_num_comments: int = 3, min_score: int = -1, years: List[int] = [2016], chunk_size: int = 10**6) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes submissions and comments for a given subreddit.
    """
    all_submissions = pd.DataFrame()
    all_comments = pd.DataFrame()

    for i, subreddit_name in enumerate(subreddit_names):
        print(f"Processing {subreddit_name:<10} ({i + 1}/{len(subreddit_names)})...")

        # Read submissions CSV file
        submissions = pd.read_csv(f"{os.path.join(data_root, subreddit_name)}_submissions.csv")

        # Add 'Y' and 'YM' columns to submissions DataFrame
        submissions = add_datetime_columns(submissions, 'created_utc')

        # Filter submissions by specified years
        submissions = submissions[submissions['Y'].isin(years)]

        # Filter submissions based on minimum score and number of comments
        submissions = submissions[submissions['score'] > min_score]
        submissions = submissions[submissions['num_comments'] >= min_num_comments]

        # Add subreddit name to DataFrame
        submissions['sub'] = subreddit_name


        # Read comments CSV file in chunks
        comments_chunks = pd.read_csv(f"{os.path.join(data_root, subreddit_name)}_comments.csv", chunksize=chunk_size)
        comments = pd.DataFrame()

        for i, chunk in enumerate(comments_chunks):
            chunk['link_id'] = chunk['link_id'].str.replace('t3_', '')

            # Keep only comments linked to the filtered submissions
            chunk = chunk[chunk['link_id'].isin(submissions['id'].unique())]

            # Remove prefix from 'parent_id'
            chunk['parent_id'] = chunk['parent_id'].str[3:]

            # Add 'Y' and 'YM' columns to comments DataFrame
            chunk = add_datetime_columns(chunk, 'created_utc')

            # Break the loop if the chunk's years are beyond the specified range
            if chunk['Y'].min() > max(years):
                break

            # Clean the 'body' text in comments
            chunk['body'] = clean_text(chunk['body'])

            # Add subreddit name to DataFrame
            chunk['sub'] = subreddit_name

            # Concatenate the processed chunk to the main comments DataFrame
            comments = pd.concat([comments, chunk], ignore_index=True)

        # Remove 't3_' prefix from 'link_id' in comments (redundant but kept for consistency)
        comments['link_id'] = comments['link_id'].str.replace('t3_', '')

        # Keep only comments linked to the filtered submissions
        comments = comments[comments['link_id'].isin(submissions['id'].unique())]

        all_submissions = pd.concat([all_submissions, submissions], ignore_index=True)
        all_comments = pd.concat([all_comments, comments], ignore_index=True)

    return all_submissions, all_comments

submissions, comments = process_subreddit(
    data_root="data/subreddits", 
    subreddit_names=["business", "climate", "energy", "labor", "education", "news"],
    min_num_comments=3,
    min_score=-1,
    years=range(2016, 2017)
)

# Filter submissions to only include those with comments
submissions = submissions[submissions['sub'].isin(comments['sub'].unique())]

print('Overall Submissions:', len(submissions))
print('Overall Comments:', len(comments))

# Get the credibility information for each domain
domain_credibility = pd.read_csv("data/domain_credibility.csv", index_col=0, header=0, names=['domain', 'bias', 'credibility'])

# Merge credibility information with submissions on domain
submissions = submissions.merge(domain_credibility, left_on='domain', right_on='domain', how='left')

# Drop submissions with missing credibility information
submissions = submissions.dropna(subset=['bias', 'credibility'])

# Remove submissions from [deleted] authors
submissions = submissions[submissions['author'] != '[deleted]']

# Calculate the average credibility rating for each author
author_credibility = submissions.groupby('author', as_index=False)['credibility'].mean()

# Remove comments from authors with no credibility information
comments = comments[comments['author'].isin(author_credibility["author"])]

print("Embedding Posts...")
# Embed the text data for comments
model = SentenceTransformer('all-MiniLM-L6-v2', device="cuda")

comments["embedding"] = comments["body"].progress_apply(lambda x: np.array(model.encode(x)))

# Calculate the average embedding for each author
author_embeddings = comments.groupby('author', as_index=False)['embedding'].apply(lambda x: np.mean(np.vstack(x), axis=0).tolist())

embed_dim = len(author_embeddings['embedding'].iloc[0])

# Calculate the number of shared comments between each pair of authors
frequencies_df = pd.crosstab(comments["author"], comments['link_id'])

# Associate credibility and embeddings with authors
credibilities = frequencies_df.merge(author_credibility, on="author", how='left')["credibility"].to_list()
embeddings = frequencies_df.merge(author_embeddings, on="author", how='left')["embedding"].to_list()
subreddits = frequencies_df.merge(comments[["author", "sub"]].groupby("author").first(), on="author", how="left")["sub"].to_list()

frequencies = np.array(frequencies_df, dtype=float)

def create_unweighted_graph(frequencies: np.ndarray, n: int = 1) -> nx.Graph:
    """
    Creates an unweighted graph from a 2D numpy array.
    """
    adjacency_matrix = frequencies @ frequencies.T
    np.fill_diagonal(adjacency_matrix, 0)

    # For unweighted graphs - connect authors with at least n shared comment(s)
    adjacency_matrix = (adjacency_matrix >= n).astype(int)

    # Create a graph from the adjacency matrix
    graph = nx.from_numpy_array(adjacency_matrix)

    # Add betweeness centrality, clustering coefficient, degree, and credibility as node attributes
    betweenness = {i: float(b) for i, b in cugraph.betweenness_centrality(graph).items()}
    nx.set_node_attributes(graph, betweenness, 'betweenness')

    clustering = {i: float(c) for i, c in cugraph.clustering(graph).items()}
    nx.set_node_attributes(graph, clustering, 'clustering')

    degree = dict(nx.degree(graph))
    nx.set_node_attributes(graph, degree, 'degree')

    credibility_dict = {i: credibility for i, credibility in enumerate(credibilities)}
    nx.set_node_attributes(graph, credibility_dict, 'credibility')

    embedding_dict = {i: {} for i in range(embed_dim)}
    for i, embedding in enumerate(embeddings):
        for j in range(embed_dim):
            embedding_dict[j][i] = embedding[j]
        
    for i in range(embed_dim):
        nx.set_node_attributes(graph, embedding_dict[i], f'embedding_{i}')

    # Add node subreddit for visualization
    subreddit_dict = {i: subreddit for i, subreddit in enumerate(subreddits)}
    nx.set_node_attributes(graph, subreddit_dict, 'subreddit')
    
    return graph

def create_weighted_graph(frequencies: np.ndarray, max_weight: int = 10) -> nx.Graph:
    """
    Creates an weighted graph from a 2D numpy array.
    """
    adjacency_matrix = frequencies @ frequencies.T
    np.fill_diagonal(adjacency_matrix, 0)

    # For weighted edges
    adjacency_matrix[adjacency_matrix > max_weight] = max_weight 

    adjacency_matrix = adjacency_matrix / adjacency_matrix.max()

    # Create a graph from the adjacency matrix
    graph = nx.from_numpy_array(adjacency_matrix)

    # Add node subreddit for visualization
    subreddit_dict = {i: subreddit for i, subreddit in enumerate(subreddits)}
    nx.set_node_attributes(graph, subreddit_dict, 'subreddit')

    # Add betweeness centrality, clustering coefficient, degree, and credibility as node attributes
    betweenness = {i: float(b) for i, b in nx.betweenness_centrality(graph, weight="weight").items()}
    nx.set_node_attributes(graph, betweenness, 'betweenness')

    clustering = {i: float(c) for i, c in nx.clustering(graph, weight="weight").items()}
    nx.set_node_attributes(graph, clustering, 'clustering')

    degree = dict(nx.degree(graph, weight="weight"))
    nx.set_node_attributes(graph, degree, 'degree')

    credibility_dict = {i: credibility for i, credibility in enumerate(credibilities)}
    nx.set_node_attributes(graph, credibility_dict, 'credibility')

    embedding_dict = {i: {} for i in range(embed_dim)}
    for i, embedding in enumerate(embeddings):
        for j in range(embed_dim):
            embedding_dict[j][i] = embedding[j]
        
    for i in range(embed_dim):
        nx.set_node_attributes(graph, embedding_dict[i], f'embedding_{i}')

    return graph

# For an unweighted graph
print("Creating unweighted graph...")
graph = create_unweighted_graph(frequencies, n=1)

nx.write_gexf(graph, "data/unweighted.gexf")

# For a weighted graph
print("Creating weighted graph, this could take a while...")
graph = create_weighted_graph(frequencies)

# Save the graph to a GEXF file
nx.write_gexf(graph, "data/weighted.gexf")


