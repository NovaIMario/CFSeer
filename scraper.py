import requests
import time
import json
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"), connect_args={"sslmode": "require"})

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}

# --- Supabase / Postgres schema (run once) ---
# CREATE TABLE problems (
#     contest_id      integer,
#     problem_index   text,
#     name            text,
#     rating          integer,
#     tags            text[],
#     time_limit      text,
#     memory_limit    text,
#     statement       text,
#     input_spec      text,
#     output_spec     text,
#     examples        jsonb,
#     note            text,
#     PRIMARY KEY (contest_id, problem_index)
# );

with engine.connect() as conn:
    conn.execute(text("SELECT 1"))


def save_problem(problem: dict):
    with engine.begin() as conn:
        result = conn.execute(
            text("""
            INSERT INTO problems
            (contest_id, problem_index, name, rating, tags,
             time_limit, memory_limit, statement, input_spec,
             output_spec, examples, note,
             editorial_url, editorial_text)
            VALUES
            (:contest_id, :problem_index, :name, :rating, :tags,
             :time_limit, :memory_limit, :statement, :input_spec,
             :output_spec, :examples, :note,
             :editorial_url, :editorial_text)
            ON CONFLICT (contest_id, problem_index) DO NOTHING
            """),
            {
                **problem,
                "tags": problem["tags"],
                "examples": json.dumps(problem["examples"]),
            },
        )
    if result.rowcount:
        print("Inserted:", problem["contest_id"], problem["problem_index"])
    else:
        print("Already exists:", problem["contest_id"], problem["problem_index"])

def get_all_problems():
    print("Fetching problem list from Codeforces API...")
    url = "https://codeforces.com/api/problemset.problems"
    response = requests.get(url, headers=HEADERS, timeout=10)
    data = response.json()
    if data["status"] != "OK":
        raise Exception("Failed to fetch problems")
    return data["result"]["problems"]


def _text_or_none(el):
    return el.get_text(" ", strip=True) if el else None

def get_editorial_url(contest_id):
    url = f"https://codeforces.com/contest/{contest_id}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        # Tutorial links usually show up in the sidebar as <a> tags
        # containing "Tutorial" or pointing to /blog/entry/
        for a in soup.find_all("a", href=True):
            if "tutorial" in a.get_text(strip=True).lower() or "editorial" in a.get_text(strip=True).lower():
                href = a["href"]
                if href.startswith("/blog/entry/"):
                    return "https://codeforces.com" + href
    except Exception as e:
        print(f"Failed to get editorial link for contest {contest_id}: {e}")
    return None


def get_editorial_text(editorial_url):
    if not editorial_url:
        return None
    try:
        resp = requests.get(editorial_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        content_div = soup.find("div", class_="ttypography") 
        if content_div:
            return content_div.get_text("\n", strip=True)
    except Exception as e:
        print(f"Failed to fetch editorial text: {e}")
    return None

def parse_statement(html_div):
    """Given the <div class="problem-statement"> tag, split it into parts."""
    header = html_div.find("div", class_="header")
    time_limit = _text_or_none(header.find("div", class_="time-limit")) if header else None
    memory_limit = _text_or_none(header.find("div", class_="memory-limit")) if header else None

    input_spec_div = html_div.find("div", class_="input-specification")
    output_spec_div = html_div.find("div", class_="output-specification")
    note_div = html_div.find("div", class_="note")
    sample_div = html_div.find("div", class_="sample-tests")

    # Main statement = every <p> that is a direct child of problem-statement,
    # sitting after the header and before input-specification.
    skip_classes = {
        "header", "input-specification", "output-specification",
        "sample-tests", "note",
    }
    statement_parts = []
    for child in html_div.find_all(recursive=False):
        child_classes = set(child.get("class", []))
        if child_classes & skip_classes:
            continue
        txt = child.get_text(" ", strip=True)
        if txt:
            statement_parts.append(txt)
    statement = "\n\n".join(statement_parts)

    input_spec = _text_or_none(input_spec_div)
    output_spec = _text_or_none(output_spec_div)
    note = _text_or_none(note_div)

    examples = []
    if sample_div:
        inputs = sample_div.find_all("div", class_="input")
        outputs = sample_div.find_all("div", class_="output")
        for inp, out in zip(inputs, outputs):
            inp_pre = inp.find("pre")
            out_pre = out.find("pre")
            in_text = inp_pre.get_text("\n", strip=True) if inp_pre else ""
            out_text = out_pre.get_text("\n", strip=True) if out_pre else ""
            examples.append({"input": in_text, "output": out_text})

    return {
        "time_limit": time_limit,
        "memory_limit": memory_limit,
        "statement": statement,
        "input_spec": input_spec,
        "output_spec": output_spec,
        "examples": examples,
        "note": note,
    }
def get_statement(contest_id, problem_index, retries=3):
    url = f"https://codeforces.com/contest/{contest_id}/problem/{problem_index}"

    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)

            if response.status_code == 429 or response.status_code == 403:
                wait = 5 * (attempt + 1)
                print(f"Rate limited on {contest_id}{problem_index}, sleeping {wait}s")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                print(f"Failed {contest_id}{problem_index}: HTTP {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            statement_div = soup.find("div", class_="problem-statement")

            if not statement_div:
                print(f"No statement found {contest_id}{problem_index}")
                return None

            parsed = parse_statement(statement_div)
            if not parsed["statement"]:
                print(f"No statement text extracted {contest_id}{problem_index}")
                return None

            return parsed

        except requests.exceptions.RequestException as e:
            print(f"  Network error on {contest_id}{problem_index} (attempt {attempt+1}/{retries}): {e}")
            time.sleep(3)

    print(f"  Giving up on {contest_id}{problem_index} after {retries} attempts")
    return None

def get_existing_ids():
    """Return set of (contest_id, problem_index) already in the DB."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT contest_id, problem_index FROM problems"))
        return set((row[0], row[1]) for row in result)

editorial_cache = {}

def collect(max_problems=100000, delay=1.0):
    problems = get_all_problems()
    problems = [p for p in problems if p.get("tags") and "contestId" in p and "index" in p]
    problems = problems[:max_problems]

    existing = get_existing_ids()
    problems = [p for p in problems if (p["contestId"], p["index"]) not in existing]
    print(f"{len(existing)} already scraped, {len(problems)} remaining")

    for i, problem in enumerate(problems):
        contest_id = problem["contestId"]
        index = problem["index"]
        tags = problem["tags"]
        rating = problem.get("rating", None)
        name = problem.get("name", "")

        if contest_id not in editorial_cache:
            ed_url = get_editorial_url(contest_id)
            ed_text = get_editorial_text(ed_url)
            editorial_cache[contest_id] = (ed_url, ed_text)
            time.sleep(delay)

        editorial_url, editorial_text = editorial_cache[contest_id]

        print(f"[{i+1}/{len(problems)}] {contest_id}{index} - {name}")
        parsed = get_statement(contest_id, index)

        if parsed:
            save_problem({
                "contest_id": contest_id,
                "problem_index": index,
                "name": name,
                "rating": rating,
                "tags": tags,
                **parsed,
                "editorial_url": editorial_url,
                "editorial_text": editorial_text,
            })
        else:
            print("  Skipping - no statement found")

        time.sleep(delay)

    print("Done!")


if __name__ == "__main__":
    collect(delay=1)