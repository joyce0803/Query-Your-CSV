import io
import pandas as pd
import streamlit as st
from lida import Manager, TextGenerationConfig , llm
import google.generativeai as genai
import os
from dotenv import load_dotenv
from PIL import Image
import base64
from lida.datamodel import Goal
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain import hub
from st_tabs import TabBar
from pandasai import SmartDataframe
from pandasai.llm.google_palm import GooglePalm
from pandasai.responses.streamlit_response import StreamlitResponse
from pandasai.helpers import path
import matplotlib.pyplot as plt
from streamlit.runtime.media_file_storage import MediaFileStorageError


load_dotenv()
st.set_page_config(layout="wide")



# api_key_llm = st.sidebar.text_input("#### Enter your API key", type="password")
# if api_key_llm:
#     # GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
#     # COHERE_API_KEY = os.getenv('COHERE_API_KEY')
#
#     lida = Manager(text_gen=llm(provider="cohere", api_key=api_key_llm))
#     textgen_config = TextGenerationConfig(temperature=0.5, model="command", use_cache=True)
#
#     genai.configure(api_key=api_key_llm)
#     llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key_llm,temperature=0,convert_system_message_to_human=True)
# else:
#     st.error("API key is required to use this application !!!!")

tools = ...
prompt = hub.pull("hwchase17/openai-functions-agent",)



# model = genai.GenerativeModel('gemini-pro')
# genai.configure(api_key=GOOGLE_API_KEY)
# pandasai_llm = GooglePalm(api_key=GOOGLE_API_KEY)



try:
    user_defined_path = path.find_project_root()
except ValueError:
    user_defined_path = os.getcwd()
user_defined_path = os.path.join(user_defined_path, "exports", "charts")

functions = [
    "boxplot",
    "describe",
    "columns",
    "columns_count",
    "corr",
    "count",
    "cov",
    "data_summarization",
    "dropna",
    "fillna",
    "filter",
    "generate_features",
    "groupby",
    "impute_missing_values",
    "max",
    "min",
    "median",
    "mean",
    "plot",
    "plot_bar_chart",
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_histogram",
    "plot_line_chart",
    "plot_pie_chart",
    "plot_scatter_chart",
    "rename",
    "rolling_mean",
    "rolling_median",
    "rolling_std",
    "rows_count",
    "std",
    "sum",
    "table_name",
    "table_description",
    "value_counts"
]

# @st.cache_data(experimental_allow_widgets=True)
def display_df():
    container1 = st.container()
    container1.success("###### First rows of your dataset ")
    container1.write(df.head(10))

# @st.cache_data(experimental_allow_widgets=True)
def data_cleaning():
    container2 = st.container()
    container2.success("###### Column Details ")
    columns_df = pandas_agent.run("What are the meaning of the columns?")
    container2.write(columns_df)
    container2.success("###### Missing Values")
    missing_values = pandas_agent.run(
        "How many missing values does this dataframe have? Start the answer with 'There are' ")
    container2.write(missing_values)
    container2.success("###### Duplicate Values")
    duplicates = pandas_agent.run("Are there any duplicate values in the dataset and if so where ?")
    container2.write(duplicates)

# @st.cache_data(experimental_allow_widgets=True)
def data_summarization():
    container3 = st.container()
    container3.success("###### Dataset Summary")
    container3.write(df.describe())
    container3.write(" ")
    container3.write(" ")
    container3.success("###### Correlation Analysis")
    correlation_analysis = pandas_agent.run(
        "Calculate correlations between numerical variables to identify potential relationships and calculate their values of all the columns.")
    container3.write(correlation_analysis)
    container3.write(" ")
    container3.write(" ")
    container3.success("###### Outliers")
    outliers = pandas_agent.run(
        "Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis. And if there are outliers how many and in which columns?")
    container3.write(outliers)
    container3.write(" ")
    container3.write(" ")
    container3.success("###### Feature Engineering")
    new_features = pandas_agent.run("What new features would be interesting to create?")
    container3.write(new_features)

# @st.cache_data(experimental_allow_widgets=True)
def function_question_variable(user_question_variable):
    try:

        if os.path.exists(os.path.join(user_defined_path, "temp_chart.png")):
            os.remove(os.path.join(user_defined_path, "temp_chart.png"))
        st.write(df)
        st.line_chart(df, y=[user_question_variable])
        summary_statistics = pandasai_agent.chat(
            f"What are the mean, median, mode, standard deviation, variance, range, quartiles, skewness and kurtosis of {user_question_variable}")
        st.write(summary_statistics)
        normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable} and plot necessary graphs")
        st.write(normality)
        try:
            response = pandasai_agent.chat(f"Check for normality or specific distribution shapes of {user_question_variable} and plot necessary graphs")
            print(response)
            st.image("exports/charts/temp_chart.png",use_column_width="auto")
        except MediaFileStorageError as m:
            pass
        outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
        st.write(outliers)
        trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable} and plot necessary graphs")
        st.write(trends)
        missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
        st.write(missing_values)
    except Exception as e:
        pass
    return

def user_queries():
    try:

        st.markdown(
            """
            <h5 style="text-align:center; ",>Ask your Queries 🤖</h5>
            """
            ,unsafe_allow_html=True
        )
        messages = st.container(height=500)

        query = st.chat_input("Enter your query")
        if query:
            with messages:
                st.chat_message("user", avatar="user.png").markdown(query)

                # Remove existing chart image
                if os.path.exists(os.path.join(user_defined_path, "temp_chart.png")):
                    os.remove(os.path.join(user_defined_path, "temp_chart.png"))

                # Process query and get response
                variable_info = pandasai_agent.chat(query=query)
                print(variable_info)

                if isinstance(variable_info, pd.DataFrame):
                    st.table(variable_info)
                if isinstance(variable_info, str):
                    if variable_info.endswith(".png") or variable_info.endswith(".jpg") or variable_info.endswith(
                            ".jpeg"):
                        image_name = os.path.basename(variable_info)
                        print(image_name)
                        st.image(Image.open(os.path.join(user_defined_path, image_name)), use_column_width="auto")
                    else:
                        response_placeholder = st.empty()
                        response_placeholder.info(variable_info)
                else:
                    response_placeholder = st.empty()
                    response_placeholder.info(variable_info)
    except Exception as e:
        st.error("Some error occurred while plotting the graph")


def function_agent():
    st.write(" ")
    component = TabBar(tabs=["Data Overview", "Data Summarization", "Analyze / Visualize", "User Queries"],default=0,color="black",activeColor="#5031F",fontSize="15px")
    if component == 0:
        col1, col2 = st.columns(2)
        with col1:
            display_df()
        with col2:
            data_cleaning()
    elif component == 1:
        data_summarization()
    elif component == 2:
        demo()
    elif component == 3:
        col1, col2 = st.columns(2)
        with col1:
            user_question_variable = st.selectbox("What variable are you interested in?", list(df.columns),index=None,placeholder='Choose an option')
            if user_question_variable is not None and user_question_variable != "":
                function_question_variable(user_question_variable)
        with col2:
            user_queries()

def demo():
    # Define the list of functions
    plot_functions = [
        "plot()",
        "plot_bar_chart(x, y)",
        "plot_confusion_matrix(y_true, y_pred)",
        "plot_correlation_heatmap()",
        "plot_histogram(column)",
        "plot_line_chart(x, y)",
        "plot_pie_chart(labels, values)",
        "plot_scatter_chart(x, y)",
        "boxplot(column)"
    ]
    functions = [
        "describe",
        "columns",
        "columns_count",
        "corr",
        "count",
        "cov",
        "data_summarization",
        "generate_features",
        "max",
        "min",
        "median",
        "mean",
        "boxplot(column)",
        "plot_bar_chart(x, y)",
        "plot_correlation_heatmap",
        "plot_histogram(column)",
        "plot_line_chart(x, y)",
        "plot_pie_chart(labels, values)",
        "plot_scatter_chart(x, y)",
        "rows_count",
        "std",
        "sum",
        "value_counts"
    ]

    # Display select box for function selection
    selected_function = st.selectbox("Select Function", functions, placeholder="Choose an option", index=None)

    # If a function is selected
    if selected_function:
        if selected_function == "columns":
            st.write(pandasai_agent.columns)
        elif selected_function == "columns_count":
            st.info(pandasai_agent.columns_count)
        elif selected_function == "plot_correlation_heatmap":
            pandasai_agent.plot_correlation_heatmap()
            plt.savefig("temp_image.png")
            st.image("temp_image.png", use_column_width='auto')
        elif selected_function == "rows_count":
            st.info(pandasai_agent.rows_count)
        else:
            function_name, *params = selected_function.split("(")
            parameters = [param.strip() for param in params]
            parameters = [param.replace(")", "") for param in parameters]

            # Get user input for parameters
            user_input = {}
            for param in parameters:
                if "," in param:
                    # Split parameters if there are multiple
                    sub_params = param.split(",")
                    for sub_param in sub_params:
                        sub_param = sub_param.strip()
                        selected_value = st.selectbox(f"Select {sub_param} for {function_name}",
                                                      [""] + list(pandasai_agent.columns), index=0)
                        user_input[sub_param] = selected_value
                        print(user_input)
                else:
                    selected_value = st.selectbox(f"Select {param} for {function_name}",
                                                  [""] + list(pandasai_agent.columns), index=0)
                    user_input[param] = selected_value


            # function call string

            function_args = ", ".join([f"'{input_val}'" for input_val in user_input.values()])
            print(function_args)
            function_call = f"df1.{function_name}({function_args})"

            # Evaluate the function call dynamically
            try:
                result = eval(function_call)
                if selected_function in plot_functions:
                    plt.savefig("temp_image.png")
                    # Display the saved image
                    st.image("temp_image.png", use_column_width='auto')
                elif isinstance(result, pd.DataFrame):
                    st.write(result)
                elif isinstance(result, str) and (result.endswith(".png") or result.endswith(".jpg") or result.endswith(".jpeg")):
                    st.image(result, use_column_width='auto')
                else:
                    st.write(result)
            except Exception as e:
                st.error(f"Error occurred while executing {selected_function}: {str(e)}")


menu = st.sidebar.selectbox('#### Choose an Option',["Summarize","Query your CSV"])
if menu == "Summarize":
    st.title("Summarization of your Data")
    api_key_lida = st.sidebar.text_input("#### Enter your API key", type="password")
    if api_key_lida:
        # GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
        # COHERE_API_KEY = os.getenv('COHERE_API_KEY')

        lida = Manager(text_gen=llm(provider="cohere", api_key=api_key_lida))
        textgen_config = TextGenerationConfig(temperature=0.2, model="command", use_cache=True)

        #### Generate summary
        summarization_methods = [
            {"label": "llm",
             "description": "Uses the LLM to generate annotate the default summary, adding details such as semantic types for columns and dataset description"},
            {"label": "default",
             "description": "Uses dataset column statistics and column names as the summary"},
            {"label": "columns", "description": "Uses the dataset column names as the summary"}]
        selected_method_label = st.sidebar.selectbox('#### Choose a summarization method',options= [method["label"] for method in summarization_methods],index=0)
        selected_method = summarization_methods[[method["label"] for method in summarization_methods].index(selected_method_label)]["label"]
        selected_method_description = summarization_methods[[method["label"] for method in summarization_methods].index(selected_method_label)]["description"]
        if selected_method:
            st.sidebar.markdown(
                f"<span> {selected_method_description} </span>", unsafe_allow_html=True
            )

        file_uploader = st.file_uploader("Upload your CSV", type="csv")
        if file_uploader is not None:
            path_to_save = "csvfile.csv"
            with open(path_to_save, "wb") as f:
                f.write(file_uploader.getvalue())

            st.write("### Summary")
            summary = lida.summarize("csvfile.csv", summary_method=selected_method, textgen_config=textgen_config)
            summaryGoal = lida.summarize("csvfile.csv", summary_method="default", textgen_config=textgen_config)
            if "dataset_description" in summary:
                st.write(summary["dataset_description"])
            if "fields" in summary:
                fields = summary["fields"]
                nfields = []
                for field in fields:
                    flatted_fields = {}
                    flatted_fields["column"] = field["column"]
                    for row in field["properties"].keys():
                        print(field["properties"])
                        if row != "samples":
                            flatted_fields[row] = field["properties"][row]
                        else:
                            flatted_fields[row] = str(field["properties"][row])
                    nfields.append(flatted_fields)
                nfields_df = pd.DataFrame(nfields)
                st.write(nfields_df)
            else:
                st.write(summary)

            ### Generate goals


            if summaryGoal:
                st.sidebar.write("#### Goal Selection")
                num_goals = st.sidebar.slider(
                    "Number of goals to generate",
                    min_value=1,
                    max_value=10,
                    value=2
                )
                own_goal = st.sidebar.checkbox("Add your own goal")
                goals = lida.goals(summaryGoal, n=num_goals, textgen_config=textgen_config)

                print(goals)
                st.write(f"### Goals ({len(goals)})")

                default_goal = goals[0].question
                goal_questions = [goal.question for goal in goals]

                if own_goal:
                    user_goal = st.sidebar.text_area("Describe your goal")
                    if user_goal:
                        new_goal = Goal(question = user_goal, visualization = str(user_goal), rationale="")

                        goals.append(new_goal)
                        goal_questions.append(new_goal.question)
                selected_goal = st.selectbox('#### Choose a generated goal',options=goal_questions,index=0)
                selected_goal_index = goal_questions.index(selected_goal)
                st.write(goals[selected_goal_index])
                selected_goal_object = goals[selected_goal_index]

                if selected_goal_object:
                    visualization_libraries = ["seaborn","matplotlib","plotly"]
                    selected_library = st.sidebar.selectbox(
                        '#### Choose a Visualization Library',
                        options=visualization_libraries,
                        index=0
                    )
                    st.write('### Visualizations')
                    num_visualizations = st.sidebar.slider(
                        "Number of visualizations ",
                        min_value=1,
                        max_value=10,
                        value=1
                    )
                    textgen_config = TextGenerationConfig(
                        n=num_visualizations,
                        temperature=0.3,
                        model="command",
                        use_cache=True
                    )
                    visualizations = lida.visualize(
                        summary=summaryGoal,
                        goal = selected_goal_object,
                        textgen_config=textgen_config,
                        library=selected_library
                    )
                    viz_titles = [f'Visualization {i+1}' for i in range(len(visualizations))]
                    selected_viz_title = st.selectbox('Choose a visualization',options=viz_titles,index=0)
                    if selected_viz_title is not None:
                        selected_viz = visualizations[viz_titles.index(selected_viz_title)]
                        if selected_viz.raster:
                            imgdata = base64.b64decode(selected_viz.raster)
                            img = Image.open(io.BytesIO(imgdata))
                            st.image(img, caption=selected_viz_title, use_column_width="auto")


                        st.write("### Visualization Code")
                        st.code(selected_viz.code)
                        print(selected_viz.code)
                        explanations = lida.explain(code=selected_viz.code, library=selected_library, textgen_config=textgen_config)
                        container = st.container(border=True)
                        for row in explanations[0]:
                            container.write(f'***{row["section"].capitalize()}***')
                            container.write(f'{row["explanation"]}')
    else:
        st.error("API key is required to use this application !!!!")

elif menu == "Query your CSV":
    st.markdown(
        """
        <div style="text-align: center;">
            <h1 style="color: dark-grey; font-weight:bold; margin-top:0px; padding-top:0px; margin-bottom:10px;">
                Query Your CSV  
                 <a href="https://emoji.gg/emoji/1938_MicrosoftExcel"><img src="https://cdn3.emoji.gg/emojis/1938_MicrosoftExcel.png" width="30px" height="30px" alt="MicrosoftExcel"></a>
            </h1>
            
        </div>

        """,
        unsafe_allow_html=True
    )
    api_key_llm = st.sidebar.text_input("#### Enter your API key", type="password")
    if api_key_llm:
        genai.configure(api_key=api_key_llm)
        # pandasai_llm = GooglePalm(api_key=api_key_llm)
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key_llm, temperature=0,convert_system_message_to_human=True,tools=tools)
        user_csv = st.sidebar.file_uploader("#### Upload your file here!", type="csv")
        if user_csv is not None:
            user_csv.seek(0)
            df = pd.read_csv(user_csv, low_memory=False)
            # df1 = SmartDataframe(df,config={"llm": llm, "response_parser": StreamlitResponse})
            pandas_agent = create_pandas_dataframe_agent(
                llm,
                df,
                agent_type="openai-tools",
                verbose=True
            )
            pandasai_agent = SmartDataframe(
                df, config={"llm":llm, "response_parser": StreamlitResponse})
            print(user_defined_path)
            st.subheader('Exploratory data analysis')

            function_agent()
    else:
        st.error("API key is required to use this application !!!!")