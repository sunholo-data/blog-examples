from sunholo.genai import GenAIFunctionProcessor
from sunholo.utils.gcp_project import get_gcp_project
import logging
from google.cloud import bigquery
import bigframes.pandas as bpd
from anthropic import AsyncAnthropicVertex, APIConnectionError, RateLimitError, APIStatusError
import traceback
import asyncio
from typing import List

# SQL creation agent
class TOMSQLCreationAgent(GenAIFunctionProcessor):
    """
    Create good SQL
    """
    def __init__(self,question, bq_project_id=None, vertex_project_id=None, region=None, credentials=None, location="EU"):

        super().__init__()

        self.project_id = bq_project_id or 'learning-ga4'
        self.vertex_project_id = vertex_project_id or get_gcp_project()
        self.region = region or "europe-west1"
        self.anthropic_client = AsyncAnthropicVertex(project_id=self.vertex_project_id, region=self.region)
        self.question = question # the question this class will create SQL for
        self.bq_client = bigquery.Client(credentials=credentials, location=location, project=self.project_id)
        logging.info(f"Creating TOMSQLCreationAgent for question: {self.question}")

    async def call_anthropic_async(self, query, temperature=0):
        try:
            logging.info(f"Calling Anthropic with {query=}")
            message = await self.anthropic_client.messages.create(
                model="claude-3-5-sonnet@20240620",
                max_tokens=8192,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            output = message.content
        except APIConnectionError as e:
            output = f"The server could not be reached {e.__cause__}"  # an underlying Exception, likely raised within httpx.
        except RateLimitError as e:
            output = f"A 429 status code was received; we should back off a bit. {str(e)}"
        except APIStatusError as e:
            output = f"Another non-200-range status code was received {e.status_code} {e.response} {traceback.format_exc()}"
        except Exception as e:
            output = f"An unknown exception was recieved: {str(e)} {traceback.format_exc()}"

        logging.info(output)
        return output

    def run_async(self, func, *args, **kwargs):
        """
        Helper function to run async methods inside sync methods using asyncio.
        """
        try:
            # Handle cases where the event loop might already be running
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return loop.run_until_complete(func(*args, **kwargs))
            else:
                return asyncio.run(func(*args, **kwargs))
        except RuntimeError:  # If no event loop is running, run a new one
            return asyncio.run(func(*args, **kwargs))
    
    def construct_tools(self) -> dict:

        def dry_run(query:str) -> dict:
            """"
            This executes a dry run on BigQuery to test that query is correct and its performance.
            """

            job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

            query_job = self.bq_client.query(query, job_config=job_config)
    
            # Wait for the dry run query to complete
            query_job.result()  # This ensures that the job is done

            # Return useful information from the dry run
            dry_run_info = {
                "total_bytes_processed": query_job.total_bytes_processed,
                "query_valid": query_job.state == "DONE",
                "errors": query_job.errors  # This will contain error messages if the query is invalid
            }

            return dry_run_info

        def generate_sql_candidates(candidates:int=10) -> List[str]:
            """
            Creates candidate SQL for the question with variations.
            This is a synchronous wrapper for the internal async version.
            """
            async def generate_sql_candidates_async(candidates=10) -> List[str]:
                tasks = [self.call_anthropic_async(self.question, temperature=1) for _ in range(candidates)]
                sql_candidates = await asyncio.gather(*tasks)
                return sql_candidates
            
            # Run the async method synchronously
            return self.run_async(generate_sql_candidates_async, candidates)
        
        def judge_best_sql(sql_candidates: List[str]) -> str:
            """
            Evaluates a list of SQL candidates and selects the best one using the Anthropic client.
            This is a synchronous wrapper for the internal async version.
            """
            async def judge_best_sql_async(sql_candidates: List[str]) -> str:
                judge_query = (
                    f"Which SQL candidate for BigQuery Google Analytics 4 export is the most likely to answer the user's question accurately?"
                    f"<question>{self.question}</question>"
                    f"<candidates>{' '.join(sql_candidates)}</candidates>"
                    "Output only the best candidate's SQL, nothing else."
                )
                best_candidate = await self.call_anthropic_async(judge_query)
                return best_candidate
            
            # Run the async method synchronously
            return self.run_async(judge_best_sql_async, sql_candidates)
        
        return {
            "dry_run": dry_run,
            "generate_sql_candidates": generate_sql_candidates,
            "judge_best_sql": judge_best_sql
        }


# Create agent
# BigQueryStudioUser, BigQuery Data View roles are good for permissions
class BerthaBigQueryAgent(GenAIFunctionProcessor):
    """
    BigQuery GA4 Agent
    """
    def __init__(self, credentials = None, project_id = None, location="EU"):
        """
        Pass in credentials object if you want to limit access to just that user
        BigQueryStudioUser, BigQuery Data View roles are good appropriate permissions
        """
        super().__init__()

        self.project_id = project_id or 'learning-ga4'
        self.client = bigquery.Client(credentials=credentials, location=location, project=self.project_id)
        bpd.options.bigquery.project = self.project_id
        bpd.options.bigquery.location = location
    
    def construct_tools(self) -> dict:

        def list_bigquery_datasets(project_id:str=None) -> list[str]:
            """
            Lists all datasets available in the connected BigQuery project.
            Often used first to see what arguments can be passed to list_bigquery_tables()
            Args:
              project_id: Not used
            """
            datasets = list(self.client.list_datasets(project=self.project_id))
            if not datasets:
                logging.info("No datasets found.")
                return []  # Return an empty list if no datasets are found
            return [dataset.dataset_id for dataset in datasets]
        
        def list_bigquery_tables(dataset_id:str) -> list[str]:
            """
            Lists all tables within a dataset.
            Args:
                dataset_id: str The name of the dataset that has tables.

            Often used after list_bigquery_datasets()

            """
            tables = list(self.client.list_tables(dataset_id))
            if not tables:
                logging.info(f"No tables found in dataset {dataset_id}.")
                return []  # Return an empty list if no tables are found
            return [table.table_id for table in tables]
        
        def get_table_schema(dataset_id: str, table_id: str) -> dict:
            """
            Retrieves the schema of a specific BigQuery table, including nested fields.

            Args:
                dataset_id: str - The BigQuery dataset ID (e.g., my_dataset_id).
                table_id: str - The BigQuery table ID.

            Returns:
                dict: A dictionary representing the schema, including nested fields.
            """
            
            def parse_field(field, prefix=""):
                """
                Recursively parse a field, including nested fields for RECORD types.

                Args:
                    field: A schema field object (e.g., google.cloud.bigquery.schema.SchemaField).
                    prefix: A string prefix for nested fields (used for nested fields).

                Returns:
                    dict: A dictionary representing the field and its nested structure.
                """
                field_name = f"{prefix}{field.name}"
                if field.field_type == "RECORD":
                    # If it's a nested field, recursively parse its subfields
                    return {field_name: {
                                "type": field.field_type,
                                "mode": field.mode,
                                "fields": {
                                    subfield.name: parse_field(subfield, prefix=f"{field_name}.")
                                    for subfield in field.fields
                                }
                            }}
                else:
                    # For non-nested fields
                    return {field_name: {"type": field.field_type, "mode": field.mode}}

            # Get the table reference and retrieve the table schema
            table_ref = self.client.dataset(dataset_id).table(table_id)
            table = self.client.get_table(table_ref)
            
            # Parse the schema, including nested fields
            schema = {}
            for field in table.schema:
                schema.update(parse_field(field))
            
            return schema
        
        def create_sql_query(question: str, table_info: str) -> dict:
            """
            Use this function to create valid SQL from the question asked.  
            It consults an expert SQL creator and should be used in most cases.
            
            Args: 
                question: str - The user's question plus other information you add to help make an accurate query.
                table_info: str - Supporting information about which table, schema, etc., that will be used to help create the correct SQL.  It must contain the relevant fields from the schema, e.g. everything needed to make a successful SQL query.
                
            Returns:
                dict: 
                    sql_workflow: str - The SQL workflow that should end with valid SQL to use downstream.
                    sql_metadata: dict - Metadata containing what functions were used to create the SQL.
            """

            # Assuming the SQL agent needs schemas as part of the content
            sql_agent = TOMSQLCreationAgent(question=question)
            the_model_name = 'gemini-1.5-pro'

            orchestrator = sql_agent.get_model(
                system_instruction=(
                    "You are a helpful SQL Creation Agent called T.O.M. "
                    f"Todays date is: {datetime.today().date()} "
                    "You are looking for the best BigQuery SQL to answer the user's question."
                    "When you think the answer has been given to the satisfaction of the user, or you think no answer is possible, or you need user confirmation or input, you MUST use the decide_to_go_on(go_on=False) function."
                    "Try to solve the problem yourself using the tools you have without asking the user, but if low likelihood of completion without, you may ask the user questions to help that will be in your chat history."
                    "If you make mistakes, attempt to fix them in the next iteration."
                    "If unsure of what exactly the user needs, take an educated guess and create an answer, but report back to the user for clarification."
                    "Do a dry run of your best candidate SQL queries to make sure they have correct syntax."
                    "Return any sql with no backticks (```) and no new line characters (e.g. \\n)"
                ),
                model_name=the_model_name
            )    

            # Create content for the SQL creation agent, passing the schemas along with the question and table info
            content = [f"Please create BigQuery SQL for this question: {question}. Here is some supporting information: {table_info}"]

            # Start the agent chat
            chat = orchestrator.start_chat()

            # Run the agent loop to generate the SQL
            agent_text, usage_metadata = sql_agent.run_agent_loop(chat, content, guardrail_max=10)

            logging.info(f"SQL agent metadata: {usage_metadata}")
            consolidator = sql_agent.get_model(
                system_instruction=(
                    "You are a helpful SQL Creation Agent called T.O.M. "
                    f"Todays date is: {datetime.today().date()} "
                    "You are looking for the best BigQuery SQL to answer the user's question."
                    "Use the generate_sql_candidates() function first to generate many candidates, then judge_best_sql() to select the best. "
                    "Only reply directly with the SQL if it is very simple - lets leverage our group of experts to help get it right! "
                    "Return any sql with no backticks (```) and no new line characters (e.g. \\n)"
                ),
                model_name=the_model_name
            )  
            response = consolidator.generate_content(f"An agent has provided the following work looking for the correct SQL.  Summarise and consolidate the results and return the best candidate SQL. {agent_text} {usage_metadata} ")

            return f"{response.text}"

        def execute_sql_query(query: str) -> bpd.DataFrame:
            """
            Executes a SQL query on BigQuery and returns the results as a BigQueryFrame.
            The function executes `import bigframes.pandas as bpd; return bpd.read_gbq(query)`
            This means 'query' can use a variety of bigframes features:
            Do not specify the project_id in your queries, that default been set for you to the correct project.
            Make sure to always include backticks ` around tablenames.

            ```python
            # read a bigquery table
            query_or_table = "ml_datasets.penguins"
            bq_df = bpd.read_gbq(query_or_table)
            # or execute SQL:
            bq_df.read_gbq("SELECT event FROM `analytics_250021309.events_20210717`")

            ```

            Args:
                query: str - The SQL query to execute, or direct files and tables
            """
            try:
                result = bpd.read_gbq(query)
                logging.info(f"{result} {type(result)}")

                return result.to_pandas().to_json(orient='records')
            
            except Exception as e:
                logging.error(f"Error executing SQL query: {str(e)}")
                raise e  # Re-raise the exception to be handled by the calling code

        
        return {
            "list_bigquery_tables": list_bigquery_tables,
            "list_bigquery_datasets": list_bigquery_datasets,
            "get_table_schema": get_table_schema,
            "execute_sql_query": execute_sql_query,
            "create_sql_query": create_sql_query
        }

# cd to tools then start via `python ppa_tool.py`
if __name__ == "__main__":
    from sunholo.utils import ConfigManager
    from sunholo.genai import init_genai
    from datetime import datetime
    init_genai()

    config=ConfigManager("bertha")
    the_model_name = 'gemini-1.5-pro-latest'
    print(f"{config=}")

    processor = BerthaBigQueryAgent()

    orchestrator = processor.get_model(
        system_instruction=(
                "You are a helpful BigQuery Agent called Bertha."
                f"Todays date is: {datetime.today().date()}"
                "You use python and BigQuery to help users gain insights from a Google Analytics 4 BigQuery export"
                "There are various bigquery tables available that contains the raw data you need to help answer user questions"
                "Use the create_sql_query to find the best SQL for the user's question.  Pass it the user question, as well as the dataset(s), table(s) and schema(s) you are looking to query"
                "Once you have the SQL, use the execute_sql_query to analyse the data to answer the questions - do not wait for user permission"
                "When you think the answer has been given to the satisfaction of the user, or you think no answer is possible, or you need user confirmation or input, you MUST use the decide_to_go_on(go_on=False) function"
                "Try to solve the problem yourself using the tools you have without asking the user, but if low likelihood of completion without you may ask the user questions to help that will be in your chat history."
                "If you make mistakes, attempt to fix them in the next iteration"
                "If unsure of what exact metrics the user needs, take an educated guess and create an answer, but report back the user they could clarify what they need "
                "If you can, provide a final output with a clean summary of results in markdown format, including data in markdown compatible tables"
            ),
        model_name=the_model_name
    )    

    content = ["Please give me the total traffic per traffic source over all dates we have available."]
    chat = orchestrator.start_chat()

    agent_text, usage_metadata = processor.run_agent_loop(chat, content, guardrail_max=10)

    print(agent_text)
    for f in usage_metadata.get('functions_called'):
        print(f"\n - {f}")


    
