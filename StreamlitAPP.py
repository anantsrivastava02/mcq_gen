import os 
import json 
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenrator.utils import read_file, get_table_data
import streamlit as st 
from langchain.callbacks import get_openai_callback
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from src.mcqgenrator.MCQGenerator import evalute_chains
from src.mcqgenrator.logger import logging


with open('C:/Users/anant/mcq_gen/Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

st.title("MCQ generation with langchain using openai")

with st.form("user_inputs"):
    uploaded_file = st.file_uploader("Upload a pdf or txt file")
    mcq_count = st.number_input("No. of MCQs" ,  min_value= 3 , max_value= 50)
    subject = st.text_input("write subject" , max_chars= 20)
    tone = st.text_input("write about complexity" , max_chars = 20 , placeholder= "Simple")
    button = st.form_submit_button("create MCQs")


    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("loading...."):
            try:
                text = read_file(uploaded_file)
                with get_openai_callback() as cb:
                    response = evalute_chains({
                        "text" : text ,
                        "number" : mcq_count,
                        "subject" : subject,
                        "tone": tone,
                        "response_json" : json.dumps(RESPONSE_JSON)
                    })
            except Exception as e:
                traceback.print_exception(type(e) , e , e.__traceback__)
                st.error("ERROR")
            else:
                print(f"Total Token :{cb.total_tokens}")
                print(f"Total prompt token :{cb.prompt_tokens}")
                print(f"Completion Token :{cb.completion_tokens}")
                print(f"Total cost :{cb.total_cost}")
                if isinstance(response , dict):
                    quiz = response.get("quiz" , None)
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index+1
                            st.table(df)
                            st.text_area(label = "Review" , value = response['review'])
                        else:
                            st.error("Error in tabe data")
                else:
                    st.write(response)