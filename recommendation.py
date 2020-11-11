"""
@Andrew

Comparrisons of two recommendation systems:
System 1: Collaborative Filtering using User Ratings
System 2: Determining distance metrics based on trail descriptions from NMF
+ Take one hike and search all hikes to find closest
--- Data is already transformed
--- search to fit conditions
--- Return 5 closest with either cosine or euclidean distance?
+ Take one hike and use NMF model to classify all the closest with the certain condition
--- Filter database on conditions set i.e. I want to stay in NY, I want to stay in NJ, I want to stay in this park
--- Transform data with NMF 
--- Return 5 closest with either cosine or euclidean distance?
--- OR --- KMeansClusters to determine similar hikes.  I.e Two hikes may be good with waterfalls and good with flowers
"""


import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import numpy as np
from copy import deepcopy


def compare_by_cosine_distance(X, y, n_to_limit=None):
    """
    Helper Function to compare one vectorized component (y)
    to a matrix of vectors (X)
    Returns the top n_to_limit closest values in X
    """
    idx = np.argsort(cosine_distances(X, y), axis=0)[:n_to_limit]
    idx = idx.reshape(1,-1)[0]
    idx_labels = X.iloc[idx].index
    return idx_labels

# Method for determining Similar hikes
def compare_hikes_by_desc(pipe, hikes_df, hike_id, n_hikes_to_show):
    """
    Searches the hikes_df for hikes that are similar by cosine distance to the given hike_id. 
    args:
        pipe (NLPPipe): pipeline object
        hikes_df (dataframe): dataframe containing all the hike information
        hike_dtm (dataframe): Document Term Matrix of the hikes_df
        n_hikes_to_show (int): Number of results to limit hikes to
    returns:
        Filtered Dataframe with hikes that are close in cosine distance to the given hike. 
    """
    if type(hike_id) == str:
        hike_id = [hike_id]
    # After everything is done we can change this to just save the transformed data in a CSV.
    hikes_dtm = pipe.vectorizer.transform(hikes_df['cleaned_descriptions'])
    hikes_df, topics = pipe.topic_transform_df(hikes_df, hikes_dtm, append_max=False)
    X = hikes_df[topics]
    y = hikes_df.loc[hike_id][topics]
    similar_hike_idx = compare_by_cosine_distance(X, y, n_hikes_to_show)
    return hikes_df.iloc[similar_hike_idx]


# method for filtering down by State/Park
def filter_hikes(hikes_df, states=None, parks=None, max_len=None, min_len=0):
    """
    Filter Hikes function to shrink dataframe down before creating recommendations.  
    args:
        hikes_df (DataFrame): Cleaned Hikes DataFrame
        states (list): list of desired states
        parks (list): list of desired parks
        max_len (int): Longest length for hike
        min_len (int): Minimum length of hike
    return:
        filtered_hikes (DataFrame): hikes_dataframe filtered by the parameters
    """
    all_masks = []
    if states:
        mask = "(hikes_df['state'].isin(states))"
        all_masks.append(mask)
    if parks:
        mask = "(hikes_df['park'].isin(parks))"
        all_masks.append(mask)
    if max_len:
        mask = "(hikes_df['trail_length'] < max_len)"
        all_masks.append(mask)
    if min_len:
        mask = "(hikes_df['trail_length'] > min_len)"
        all_masks.append(mask)

    if all_masks:
        all_masks = " & ".join(all_masks)
        filtered_hikes = deepcopy(hikes_df[eval(all_masks)])
    else:
        print('Warning! No Filters Applied to Dataframe')
        filtered_hikes = deepcopy(hikes_df)
    
    return filtered_hikes

def get_top_3_tags(hike):
    tags = ['parking', 'rock', 'ice', 'lake', 'easy', 'hard',
       'bug', 'family', 'maintain']
    tags = ['Parking Issues', 'Rocky Conditions', 'Snow/Icy Conditions',
       'Lake/Waterfall/Pond', 'Easy Difficulty', 'Hard Difficulty',
       'Bring Bug Spray', 'Family Friendly', 'Well Maintained']
    idx = np.argsort(hike)[:-4:-1]
    toptags = ', '.join([tags[i] for i in idx])
    return toptags


def comparrison(comp_id, ht_mat, dt_mat, r_lim=50, desc_lim=10):
    """
    Finds the 50 most similar hikes in the reviews Hike-Topic Matrix.
    Filters the Trail Description Topic Matrix to the top 10 most similar hikes
    args: 
        comp_id(str): hike_id to compare
        ht_mat(dataframe): Hike-Topic Matrix Aggregated by Review
        dt_mat(dataframe): Hike-Topic Matrix Aggregated by Description
        r_lim(int): Primary limit on how many similarly reviewed hikes to consider
        desc_lim(int): Limit on how may hikes to show by similar descriptions
    Returns the indexes of the hikes from the Hike_df

    ### Important: DO NOT SORT MATRIXIES.  the ht mat and hike_df should have the same indexes
    """
    sim_idx = compare_by_cosine_distance(y=ht_mat.loc[[comp_id]], X=ht_mat, n_to_limit=r_lim) # 
    comp_dt = dt_mat.loc[[comp_id]] # Create Target Description Topic Matrix
    sim_dt = dt_mat.loc[sim_idx] # Filter Description Topic Matrix to similarly reviewed hikes
    dt_idx = compare_by_cosine_distance(y=comp_dt, X=sim_dt, n_to_limit=desc_lim) # get indexes of top 10 similar hikes by descriptions
    return dt_idx

if __name__ == "__main__":
    from nlp import NLPPipe

    # load hike info dataframe 
    hikes_df = pd.read_csv('../src/clean_all_hikes.csv', index_col=0)
    hikes_df.set_index('hike_id', inplace=True) 

    # Load Pipe
    pipe = NLPPipe()
    pipe.load_pipe(filename='../models/nmf_trail_desc.mdl')

    hikes_dtm = pipe.vectorizer.transform(hikes_df['cleaned_descriptions'])
    hikes_df, topics = pipe.topic_transform_df(hikes_df, hikes_dtm, append_max=False)

    # Description-Topic Matrix
    dt_mat = hikes_df[topics]

    hikes_df.drop(columns=topics, inplace=True)

    # Prepare Hike-Tag-Matrix
    r_corex_df = pd.read_csv('../src/reviews_corex.csv', index_col=0)
    tags = ['parking', 'rock', 'ice', 'lake', 'easy', 'hard',
       'bug', 'family', 'maintain']
    
    # Rounds tags to 1 for topic importance   
    r_corex_df[['parking', 'rock', 'ice', 'lake', 'easy', 'hard', 'bug', 'family', 'maintain']] = r_corex_df[['parking', 'rock', 'ice', 'lake', 'easy', 'hard', 'bug', 'family', 'maintain']].round(1)
    # hike tag matrix
    ht_mat = r_corex_df.groupby('hike_id')['parking', 'rock', 'ice', 'lake', 'easy', 'hard', 'bug', 'family', 'maintain'].sum()
    
    # Rename Columns for Better Tags
    col_dict = {
        'parking': 'Parking Issues',
        'rock': 'Rocky Conditions', 
        'ice': 'Snow/Icy Conditions', 
        'lake': 'Lake/Waterfall/Pond', 
        'easy': 'Easy Difficulty',
        'hard': 'Hard Difficulty',
        'bug': 'Bring Bug Spray',
        'family': 'Family Friendly',
        'maintain': 'Well Maintained'}
    ht_mat.rename(columns=col_dict, inplace=True)
    r_corex_df.rename(columns=col_dict, inplace=True)

    # Append top 3 tags to hikes_df    
    tag_col = []
    for _, row in ht_mat.iterrows():
        tag_col.append(get_top_3_tags(row))
    ht_mat['temp_tag'] = tag_col

    # Save top tags to hikes_df
    hikes_df = hikes_df.merge(ht_mat['temp_tag'], left_index=True, right_index=True, how='left')
    # Save dummy top terms as csv for filtering by tags
    hike_tag_dummies =  hikes_df.temp_tag.str.get_dummies(sep=', ')

    # drop temp tags from htmat
    ht_mat.drop(columns='temp_tag', inplace=True)
    
    dt_mat.to_csv('dt_mat.csv')
    hike_tag_dummies.to_csv('hike_tag_dummies.csv')
    ht_mat.to_csv('ht_mat.csv')
    hikes_df.to_csv('hikes_df.csv')
    r_corex_df.to_csv('r_corex.csv')    

