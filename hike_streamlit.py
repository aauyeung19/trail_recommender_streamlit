import streamlit as st
import pandas as pd
import numpy as np
from recommendation import comparrison, filter_hikes 

@st.cache
def load_tables():
    hikes_df = pd.read_csv('hikes_df.csv', index_col=0)
    # hikes_df = hikes_df[hikes_df['trail_length']<100]
    hike_tag_dummies = pd.read_csv('hike_tag_dummies.csv', index_col=0)
    ht_mat = pd.read_csv('ht_mat.csv', index_col=0)
    dt_mat = pd.read_csv('dt_mat.csv', index_col=0)
    reviews = []
    r_list = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9']
    for filename in r_list:
        reviews.append(pd.read_csv(filename+'.csv', index_col=0))
    reviews = pd.concat(reviews, ignore_index=True)
    return hikes_df, hike_tag_dummies, ht_mat, dt_mat, reviews

@st.cache(allow_output_mutation=True)
def store_random_id():
    return []

@st.cache(allow_output_mutation=True)
def store_comp_id():
    return []

if __name__ == "__main__":
    hikes_df, hike_tag_dummies, ht_mat, dt_mat, reviews = load_tables()
    cached_id = store_comp_id()

    locked_title = st.sidebar.empty()
    locked_title_value = st.sidebar.empty()
    locked_info = st.sidebar.empty()
    nav = st.sidebar.radio("Navigation", ['Select a Hike', 'Compare Hikes'])
    if nav == 'Select a Hike':
        st.title("All Trails Recommendations")
        st.image('mt washington ravine .JPG', use_column_width=True)
        st.markdown(
            "Hello!  Please feel feel free to use this mini app to find recommendations for hikes you enjoy!"
            "When you've selected a hike you like, be sure to **lock in** your selection at the bottom"
        )
        method = st.radio('Pick your method', ["Search By Trail Name", "Select a Hike", "I'm Feeling Lucky"])
        if method == "Search By Trail Name":
            trail_name = st.selectbox('Type your search here:', hikes_df["trail_name"].unique())
            if not trail_name:
                st.stop()
            mask = hikes_df["trail_name"] == trail_name
            comp_id = hikes_df[mask].index


        if method == "Select a Hike":
            st.write("What hike did you like?")
            state = st.selectbox('State', hikes_df["state"].unique())
            mask = (hikes_df["state"] == state)
            park = st.selectbox('Park', hikes_df[mask]['park'].unique())
            mask = mask & (hikes_df["park"] == park)
            trail_name = st.selectbox('Trail Name', hikes_df[mask]["trail_name"].unique())
            mask = mask & (hikes_df["trail_name"] == trail_name)
            comp_id = hikes_df[mask].index

        if method == "I'm Feeling Lucky":
            random_id_cache = store_random_id()
            if len(random_id_cache) == 0:
                st.balloons()
                random_id_cache.append(np.random.choice(hikes_df.index))
            comp_id = hikes_df.loc[[random_id_cache[-1]]].index

            clear_cache = st.button('Show me another', False)
            if clear_cache:
                # random_id_cache.clear()
                random_id_cache.append(np.random.choice(hikes_df.index))
                comp_id = hikes_df.loc[[random_id_cache[-1]]].index
                clear_cache = False

            st.subheader("Trail Name")
            st.write(hikes_df.loc[comp_id]["trail_name"][0])
            st.subheader("State")
            st.write(hikes_df.loc[comp_id]["state"][0])
            st.subheader("Park")
            st.write(hikes_df.loc[comp_id]["park"][0])

        # Show Selection Descriptions
        st.subheader("Trail Description")
        st.write(hikes_df.loc[comp_id]["trail_description"][0])
        st.write("Review Tags: ", hikes_df.loc[comp_id]["temp_tag"][0])
        st.write("Link: ", hikes_df.loc[comp_id]["link"][0])

        # Show reviews that talk about tags
        tag = st.selectbox('See Reviews that talk about: ', ['Parking Issues', 'Rocky Conditions',
        'Snow/Icy Conditions', 'Lake/Waterfall/Pond', 'Easy Difficulty',
        'Hard Difficulty', 'Bring Bug Spray', 'Family Friendly',
        'Well Maintained'])
        
        # Filter Hikes by Tags
        filtered_reviews = reviews[reviews['hike_id']==comp_id[0]]
        st.table(
            filtered_reviews[['date','user_desc']].iloc[filtered_reviews[tag].argsort()[:-4:-1]].set_index('date')
        )
    
        # Lock in ID for nav
        lock = st.button('Lock In Hike', False)
        if lock:
            cached_id.append(comp_id)

    try:
        locked_title.subheader("Trail Name: ")
        locked_title_value.write(str(hikes_df.loc[cached_id[-1]]['trail_name'][0]))
        locked_info.write(str(hikes_df.loc[cached_id[-1]]['trail_description'][0]))
    except:
        if len(cached_id) == 0:
            st.warning("Don't forget to Lock In your selection!")
            st.stop()
    if nav == "Compare Hikes":
        st.image('header.JPG', use_column_width=True)
        comp_id = hikes_df.loc[[cached_id[-1][0]]].index
        st.header("Filter Your Hikes")
        state_filter = st.multiselect('State', hikes_df["state"].unique())
        mask = (hikes_df["state"].isin(state_filter))
        park_filter = st.multiselect('Park', hikes_df[mask]['park'].unique())
        min_len_filter = st.slider('Minimum Trail Length', 0, 400)
        max_len_filter = st.slider('Maximum Trail Length', min_len_filter+1, 400, 400)
        
        filtered_indexes = filter_hikes(
            hikes_df, 
            states=state_filter, 
            parks=park_filter,
            max_len=max_len_filter,
            min_len=min_len_filter).index
        filtered_indexes = filtered_indexes.append(comp_id)
        ht_mat = ht_mat.loc[filtered_indexes]
        dt_mat = dt_mat.loc[filtered_indexes]
        ht_mat.drop_duplicates(inplace=True)
        dt_mat.drop_duplicates(inplace=True)

        show = st.checkbox("Show Me Similar Hikes!", False)

        if not ht_mat.shape[0]:
            st.error('No Trails Exist with the Filtered Conditions.')
        
        if not show:    
            st.stop()

        sim_idx = comparrison(comp_id=comp_id[0], ht_mat=ht_mat, dt_mat=dt_mat, r_lim=30, desc_lim=3)
        if len(sim_idx) == 1:
            st.error('No Trails Exist with the Filtered Conditions.')
        for n, i in enumerate(sim_idx[1:], start=1):
            st.header(f"Recommendation {n}")
            st.subheader("Trail Name")
            st.write(hikes_df.loc[i]["trail_name"])
            st.subheader("State")
            st.write(hikes_df.loc[i]["state"])
            st.subheader("Park")
            st.write(hikes_df.loc[i]["park"])
            st.subheader("Trail Description")
            st.write(hikes_df.loc[i]["trail_description"])
            st.write("Review Tags: ", hikes_df.loc[i]["temp_tag"])
            st.write("Link: ", hikes_df.loc[i]["link"])
            st.markdown("-------------------------------------")
        