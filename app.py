# Copyright @Siddharth

import os
import pandas as pd
import streamlit as st
import random
import time

from api_key import apikey
from langchain_openai import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from dotenv import load_dotenv, find_dotenv


from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
#from langchain.agents.agent_toolkits import create_python_agent
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
# from langchain.utilities import WikipediaAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper


# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory


from streamlit_chat import message
# from conversation import *


#Caching
@st.cache_data
def eda_steps(_llm):
    steps = _llm('What are the steps for EDA')
    return steps

@st.cache_data
def algorithm_selection(_llm):
    data_science_framing = _llm("Write a couple of paragraphs about the importance of considering more than one algorithm when trying to solve a data science problem")
    return data_science_framing



@st.cache_data
def eda_function(_pandas_agent):
    st.write("**Data Overview**")
    columns_df = _pandas_agent.run("What are the meaning of the columns?")
    st.write(columns_df)
    missing_values = _pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
    st.write(missing_values)
    duplicates = _pandas_agent.run("Are there any duplicate values and if so where?")
    st.write(duplicates)
    st.write("**Data Summarisation**")
    # correlation_analysis = _pandas_agent.run("Calculate correlations between variables using df.corr() function to identify potential relationships.")
    # st.write(correlation_analysis)
    outliers = _pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
    st.write(outliers)
    new_features = _pandas_agent.run("What new features would be interesting to create?.")
    st.write(new_features)

    return

# Define functions for each button
@st.cache_data
def perform_eda(_llm,_pandas_agent):
    with st.sidebar:
        with st.expander("**What are the steps for EDA**"):
            st.write(eda_steps(_llm))
    st.write("Performing EDA...")
    eda_function(_pandas_agent)
    

@st.cache_data
def function_question_variable(df,user_question_variable,_pandas_agent):
    # st.line_chart(df, y = [user_question_variable])
    summary_statistics = _pandas_agent.run(f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of Close")
    st.write(summary_statistics)
    normality = _pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
    st.write(normality)
    outliers = _pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
    st.write(outliers)
    trends = _pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
    st.write(trends)
    # missing_values = _pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
    # st.write(missing_values)
    return

@st.cache_resource
def wiki(prompt):
    wiki_research = WikipediaAPIWrapper().run(prompt)
    return wiki_research

@st.cache_data
def prompt_templates():
    data_problem_template = PromptTemplate(
                            input_variables=["business_problem"],
                            template="Covert the following business problem: {business_problem} into a data science problem"
                            )
    
    model_selection_template = PromptTemplate(
                            input_variables=["data_problem"],
                            template="Give me a list of machine learning algorithms for this problem: {data_problem}, while using this wikipedia research: {wikipedia_research}"
                            )

    return data_problem_template, model_selection_template

@st.cache_resource
def chains(_llm):
    data_problem_template, model_selection_template = prompt_templates()
    data_problem_chain = LLMChain(llm = _llm, prompt = data_problem_template, verbose = True, output_key = 'data_problem')
    model_selection_chain = LLMChain(llm = _llm, prompt = model_selection_template, verbose = True, output_key = 'model_selection')
    sequential_chain = SequentialChain(chains = [data_problem_chain,model_selection_chain],
                                                        input_variables = ['business_problem','wikipedia_research'], 
                                                        output_variables = ['data_problem','model_selection'],
                                                        verbose = True)
    
    return sequential_chain

@st.cache_data
def chains_output(prompt, wiki_research, _llm):
    my_chain = chains(_llm)
    my_chain_output = my_chain({'business_problem': prompt, 'wikipedia_research': wiki_research})
    my_data_problem = my_chain_output["data_problem"]
    my_model_selection = my_chain_output["model_selection"]

    return my_data_problem, my_model_selection

@st.cache_data
def list_to_selectbox(my_model_selection_input):
    algorithm_lines = my_model_selection_input.split('\n')
    algorithms = [algorithm.split(':')[-1].split('.')[-1].strip() for algorithm in algorithm_lines if algorithm.strip()]
    algorithms.insert(0, "Select Algorithm")
    formatted_list_output = [f"{algorithm}" for algorithm in algorithms if algorithm]
    return formatted_list_output


@st.cache_data
def python_solution(my_data_problem, selected_algorithm, user_csv, _python_agent):
    solution = _python_agent.run(
        f"Write a Python script to solve this: {my_data_problem}, using this algorithm: {selected_algorithm}, using this as your dataset: {user_csv}."
        )
    return solution

@st.cache_data
def perform_data_science(prompt, _llm, user_csv, _python_agent):
    wiki_research = wiki(prompt)
    my_data_problem = chains_output(prompt,wiki_research, _llm)[0]
    my_model_selection = chains_output(prompt,wiki_research, _llm)[1]

    st.write(my_data_problem)
    st.write(my_model_selection)

    with st.sidebar:
        with st.expander("**Is one algorithm enough?**"):
            st.caption(algorithm_selection(_llm))
    

    formatted_list = list_to_selectbox(my_model_selection)

    # selected_algorithm = st.selectbox("Select Machine Learning Algorithm", formatted_list)

    # if "selected_algorithm" not in st.session_state:
    #     st.session_state.selected_algorithm = "Select Algorithm"

    # if st.session_state.selected_algorithm != selected_algorithm:
    #     st.session_state.selected_algorithm = selected_algorithm

    # if selected_algorithm != "Select Algorithm":
    #     st.subheader("Solution")
    #     solution = python_solution(my_data_problem, selected_algorithm, user_csv, _python_agent)
    #     st.write(solution)


def main():
    os.environ['OPENAI_API_KEY'] = apikey
    load_dotenv(find_dotenv())

    #Title
    st.title('Intellidata: AI Assisted Data Science Platform 🤖')

    # Welcoming Message
    st.write("Hello! 👋 I am your AI assitance, here to help you with your Data Science and Machine Learning Project")

    #Sidebar
    with st.sidebar:
        st.write("*Your Data Science Experience begins with uploading a csv file!*")
        st.caption("""**Please upload a csv file so that we can get started.
                Once we have the data, we'll dive into understanding the data and how we can use it to tackle your business problem.
                I also can introduce you to cool machine learning models that you can use to tackle your problem**""")
        
        st.divider()

    #Initialise the key in session
    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1:False}

    #Function to update the value in session state
    def clicked(button):
        st.session_state.clicked[button] = True

    st.button("Let's get started",on_click = clicked, args=[1])

    if st.session_state.clicked[1]:
        st.subheader('Your first step towards Data Science Projects start with Understanding Data.')

        user_csv = st.file_uploader("Upload your file here",type = 'csv')

        if user_csv is not None:
            #Making sure the cursor is at the start of the file
            user_csv.seek(0)

            #Converting the csv file to a dataframe
            df = pd.read_csv(user_csv,low_memory=False)
            

            #LLM model load
            llm = OpenAI(temperature=0)

            #Pandas agent
            # pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

            pandas_agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
                handle_parsing_errors=True,
                allow_dangerous_code=True
                )

            # @st.cache_resource
            # def python_agent():
            #     agent_executor = create_python_agent(
            #         llm=llm,
            #         tool=PythonREPLTool(),
            #         verbose=True,
            #         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            #         handle_parsing_errors=True,
            #         )
            #     return agent_executor
            

            python_agent = create_python_agent(
                    llm=llm,
                    tool=PythonREPLTool(),
                    verbose=True,
                    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    handle_parsing_errors=True,
                    )
            
            # Initialize session state
            if 'button_state' not in st.session_state:
                st.session_state.button_state = {
                    "EDA": False,
                    "Data Analysis": False,
                    "Data Science": False
                }

            # Create buttons
            if st.button("EDA"):
                st.session_state.button_state["EDA"] = True
                perform_eda(llm,pandas_agent)

            
            if st.button("Data Analysis"):
                st.session_state.button_state["Data Analysis"] = True

            if st.session_state.button_state.get("Data Analysis", False):
                with st.form(key='data_analysis_form'):
                    user_question_variable = st.text_input("What variable are you interested in?")
                    submit_button = st.form_submit_button(label='Submit')

                if submit_button and user_question_variable:
                    with st.sidebar:
                        with st.expander("**What are the steps for Data Analysis**"):
                            st.write(llm('What are the steps for EDA'))
                    function_question_variable(df,user_question_variable, pandas_agent)

            if st.button("Data Science"):
                st.session_state.button_state["Data Science"] = True
                

            if st.session_state.button_state.get("Data Science", False):
                with st.form(key='data_science_form'):
                    prompt = st.text_area("What is the business problem you would like to solve?")
                    submit_button = st.form_submit_button(label='Submit')

                if submit_button and prompt:
                    perform_data_science(prompt, llm, user_csv, python_agent)

            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Display chat messages from history on app rerun
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # # Accept user input
            if prompt := st.chat_input("What is up?"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Display assistant response in chat message container
                with st.chat_message("assistant"):
                    response = pandas_agent.run(prompt)
                    st.markdown(response)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == '__main__':
    main()