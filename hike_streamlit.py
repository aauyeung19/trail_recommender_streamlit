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
    return hikes_df, hike_tag_dummies, ht_mat, dt_mat

@st.cache
def save_comp_id(comp_id):
    return (comp_id)
if __name__ == "__main__":
    hikes_df, hike_tag_dummies, ht_mat, dt_mat = load_tables()

    st.title("All Trails Recommendations")
    method = st.sidebar.radio('Pick your method', ["Select a Hike", "I'm Feeling Lucky"])
    if method == "Select a Hike":
        st.write("What hike did you like?")
        state = st.selectbox('State', hikes_df["state"].unique())
        mask = (hikes_df["state"] == state)
        park = st.selectbox('Park', hikes_df[mask]['park'].unique())
        mask = mask & (hikes_df["park"] == park)
        trail_name = st.selectbox('Trail Name', hikes_df[mask]["trail_name"].unique())
        mask = mask & (hikes_df["trail_name"] == trail_name)
        comp_id = hikes_df[mask].index[0]
        st.button('Show Me')
    else:
        st.balloons()
        comp_id = np.random.choice(hikes_df.index)
        st.subheader("Trail Name")
        st.write(hikes_df.loc[comp_id]["trail_name"])
        st.subheader("State")
        st.write(hikes_df.loc[comp_id]["state"])
        st.subheader("Park")
        st.write(hikes_df.loc[comp_id]["park"])
    st.subheader("Trail Description")
    st.write(hikes_df.loc[comp_id]["trail_description"])
    st.write("Review Tags: ", hikes_df.loc[comp_id]["temp_tag"])

    show = st.button("Show Me Similar Hikes!", True)

    if not show:
        
        st.stop()

    sim_idx = comparrison(comp_id=comp_id, ht_mat=ht_mat, dt_mat=dt_mat, r_lim=30, desc_lim=3)

    for n, i in enumerate(sim_idx[1:], start=1):
        st.write(f"Recommendation {n}")
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
        