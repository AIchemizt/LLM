import streamlit as st
import langchain_helper

st.title("Restaurant Name Generator")

cuisine = st.sidebar.selectbox("Pick a Cusine", ("Japan", "Italian", "Mexican", "Arabic", "American"))


if cuisine:
    response = langchain_helper.generate_restaurant_info(cuisine)

    st.header(response["restaurant_name"])
    menu_items = response["menu_items"].split(",")

    st.write("Menu Items")
    for item in menu_items:
        st.write("-", item)