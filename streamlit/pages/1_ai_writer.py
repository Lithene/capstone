import streamlit as st
import pandas as pd

import google.generativeai as palm
import helper_streamlit as helper

st.set_page_config(page_title="AI Writer", page_icon="📑")
st.title('AI Writer')
st.subheader('Welcome to the AI writer!')
input_text = st.text_area('Write your post below so that the writer provide a compliancy check and provide suggestions to improve on the post.', 'Insert draft here...')

if (input_text != 'Insert draft here...') and (len(input_text) != 0):

    st.write(f'You wrote {len(input_text)} characters.')

    # Call Palm API
    helper.config_palm()
    # Configure the model and instructions
    text, defaults = helper.setup_palm()

    # Configure question into the model
    question = text + input_text

    # Call the model and print the response.
    response = palm.generate_text(**defaults,
                                prompt=question
                                )
    st.write(f"{response.result}")