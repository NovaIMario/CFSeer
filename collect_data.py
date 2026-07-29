import requests
import json
import time
from bs4 import BeautifulSoup
"""
Loading problems.json...
Loaded 10118 problems
F1 Score (micro): 0.299
F1 Score (macro): 0.175
Saving model...
Done!
"""
 
def get_all_problems():
    print("Fetching problem list from Codeforces API...")
    url = "https://codeforces.com/api/problemset.problems"
    response = requests.get(url)
    data = response.json()
    if data["status"] != "OK":
        raise Exception("Failed to fetch problems")
    return data["result"]["problems"]

def get_statement(contest_id, problem_index):
    url = f"https://codeforces.com/contest/{contest_id}/problem/{problem_index}"
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        statement = soup.find("div", class_="problem-statement")
        for div in statement.find_all("div", class_=["input-specification", "output-specification", "note", "sample-tests"]):
            div.decompose()
        return statement.get_text(separator=" ").strip()
    except Exception as e:
        print(f"Failed {contest_id}{problem_index}: {e}")
    return None

def collect(max_problems=100000, delay = 0.3):
    problems = get_all_problems()
    problems = [p for p in problems if p.get("tags") and "contestId" in p and "index" in p]
    problems = problems[:max_problems]
    results = []
    for i, problem in enumerate(problems):
        contest_id = problem["contestId"]
        index = problem["index"]
        tags = problem["tags"]
        rating = problem.get("rating", None)
        name = problem.get("name", "")
        print(f"[{i+1}/{len(problems)}] {contest_id}{index} - {name}")
        statement = get_statement(contest_id, index)
        if statement:
            results.append({
                "contest_id": contest_id,
                "index": index,
                "name": name,
                "rating": rating,
                "tags": tags,
                "statement": statement
            })
        else:
            print(f"  Skipping - no statement found")
        if (i + 1) % 100 == 0:
            with open("problems.json", "w") as f:
                json.dump(results, f)
            print(f"Saved {len(results)} so far")
        time.sleep(delay)  
    with open("problems.json", "w") as f:
        json.dump(results, f)
    print(f"Done!")

if __name__ == "__main__":
    collect(delay=0.3)