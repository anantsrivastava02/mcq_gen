import os
import json 
import traceback
import pandas as pd
from dotenv import load_dotenv
from mcqgenrator.logger import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain 

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key = key , model_name = 'gpt-3.5-turbo', temperature = 0.3)
Template = """
Text : {text}
you are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice question for {subject} students in {tone} tone.
Make sure the question are not repeated and check all the question to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide . \
Ensure to make {number} MCQs

{response_json} 
"""


prompt1 = PromptTemplate(
    input_variables = ["text" , "number" ,"subject" , "tone" , "response_json"],
    template = Template ,
) 

quiz_chain = LLMChain(llm = llm , prompt = prompt1 , output_key = "quiz" , verbose  = True)

Template2 = """
you are an expert english grammarian and writer. Given a Multiple Choice quiz for {subject} students.\
you need to evalute the complexity of the question and give a complete analysis of the quiz. Only use it max 50 words for complexity
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz question which needs to be changed and change the tone such that it perfectly fits the students ability
Quiz_MCQs :{quiz} 
check for an expert English Writer of the above quiz:
"""

quiz_ev_prompt = PromptTemplate(
    input_variables = ["subject" , "quiz"] ,
    template = Template2,
)

review_chain = LLMChain(llm = llm , prompt = quiz_ev_prompt , output_key= "review" , verbose=True)

evalute_chains = SequentialChain(
    chains=[quiz_chain, review_chain], 
    input_variables=["text", "number", "subject", "tone", "response_json"],
    output_variables=["quiz", "review"],
    verbose=True
)


