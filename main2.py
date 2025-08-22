from futurehouse_client import FutureHouseClient, JobNames
from futurehouse_client.models import (
    RuntimeConfig,
    TaskRequest,
)
from ldp.agent import AgentConfig
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables")

client = FutureHouseClient(
    api_key=api_key,
)

task_data = TaskRequest(
    name=JobNames.from_string("crow"),
    query="What is the molecule known to have the greatest solubility in water?",
)
responses = client.run_tasks_until_done(task_data)
task_response = responses[0]

print(f"Job status: {task_response.status}")
print(f"Job answer: \n{task_response.formatted_answer}")

agent = AgentConfig(
    agent_type="SimpleAgent",
    agent_kwargs={
        "model": "gpt-4o",
        "temperature": 0.0,
    },
)
task_data = TaskRequest(
    name=JobNames.CROW,
    query="How many moons does earth have?",
    runtime_config=RuntimeConfig(agent=agent, max_steps=10),
)
responses = client.run_tasks_until_done(task_data)
task_response = responses[0]

print(f"Job status: {task_response.status}")
print(f"Job answer: \n{task_response.formatted_answer}")

task_data = TaskRequest(
    name=JobNames.CROW, query="How many species of birds are there?"
)

responses = client.run_tasks_until_done(task_data)
task_response = responses[0]

print(f"First job status: {task_response.status}")
print(f"First job answer: \n{task_response.formatted_answer}")

continued_job_data = {
    "name": JobNames.CROW,
    "query": "From the previous answer, specifically,how many species of crows are there?",
    "runtime_config": {"continued_job_id": task_response.task_id},
}

responses = client.run_tasks_until_done(continued_job_data)
continued_task_response = responses[0]


print(f"Continued job status: {continued_task_response.status}")
print(f"Continued job answer: \n{continued_task_response.formatted_answer}")