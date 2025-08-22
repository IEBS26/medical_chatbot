from futurehouse_client import FutureHouseClient, JobNames
from futurehouse_client.models import TaskRequest
import os
from dotenv import load_dotenv
import time
# Load API key
load_dotenv()
api_key = os.getenv("API_KEY5")
if not api_key:
    raise ValueError("API_KEY not found in environment variables")

# Initialize client
client = FutureHouseClient(api_key=api_key)

def chat_loop():
    print("🤖 Context-aware chatbot ready! (type 'no' or 'exit' to stop)")
    continued_job_id = None  # store last task_id for context
    
    while True:
        user_query = input("You: ")
        start = time.time()
        if user_query.lower() in ["no", "exit", "quit"]:
            print("👋 Goodbye!")
            break
  
        # Prepare request 
        task_data = TaskRequest(
            name=JobNames.CROW,  # or any other job name you need
            query=user_query,
            runtime_config={"continued_job_id": continued_job_id} if continued_job_id else None
        )

        # Run the task
        responses = client.run_tasks_until_done(task_data)
        task_response = responses[0]

        # Print answer
        print(f"Bot: {task_response.formatted_answer}")
        end = time.time()
        print(f"Time taken for the query: {end - start:.2f} seconds")
        # Save task_id for context in next round
        continued_job_id = task_response.task_id

# Run chatbot
if __name__ == "__main__":
    chat_loop()
 