from futurehouse_client import FutureHouseClient, JobNames
from futurehouse_client.models.app import TaskRequest
from dotenv import load_dotenv
import os
import time
from colorama import Fore, Style, init
init(autoreset=True)
# Load API key
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not found in environment variables")

# Initialize client
client = FutureHouseClient(api_key=api_key)

# Model selection


print("\n" + Fore.CYAN + "Select a model to run your query:\n")
print(Fore.YELLOW + "1. Crow     " + Style.RESET_ALL + "- Concise search")
print(Fore.YELLOW + "2. Falcon   " + Style.RESET_ALL + "- Deep Research")
print(Fore.YELLOW + "3. Phoenix  " + Style.RESET_ALL + "- Chemistry Tasks (Experimental)")
print(Fore.YELLOW + "4. Owl      " + Style.RESET_ALL + "- Precedental Search")

choice = input(Fore.GREEN + "\nEnter your choice (1-4): " + Style.RESET_ALL).strip()

if choice == "1":
    job_name = JobNames.CROW
elif choice == "2":
    job_name = JobNames.FALCON
elif choice == "3":
    job_name = JobNames.PHOENIX
elif choice == "4":
    job_name = JobNames.OWL
else:
    raise ValueError(Fore.RED + "❌ Invalid choice. Please select between 1 and 4.")

# First query
user_query = input("Enter your query: ")
start =time.time()
print("Processing your query...")
task_request = TaskRequest(
    name=job_name,
    query=user_query,
)

task_responses = client.run_tasks_until_done(
    TaskRequest(
        name=job_name,
        query=user_query,
    ),
    verbose=True,
)

print("\n=== Initial Response ===")
print(task_responses.answer)
end = time.time()
print(Fore.GREEN + f"Time taken: {end - start:.2f} seconds")
# Save task ID
task_id = client.create_task(task_responses)

# # Follow-up query
# follow_up_query = input("\nDo you want to ask a follow-up question? (yes/no): ")
# if follow_up_query.lower() == "yes":
#     follow_up_question = input("Enter your follow-up question: ")
    
#     continued_task_data = TaskRequest(
#         name=job_name,
#         query=follow_up_question,
#         runtime_config={"continued_job_id": task_responses[0].id},  # use first task's id
#     )
    
#     follow_up_responses = client.run_tasks_until_done(continued_task_data, verbose=True)
    
#     print("\n=== Follow-up Response ===")
#     for i, resp in enumerate(follow_up_responses, start=1):
#         print(f"\n--- Result {i} ---")
#         for r in resp.results:
#             print("Answer:", r.answer)
#             print("Metadata:", r.metadata)
