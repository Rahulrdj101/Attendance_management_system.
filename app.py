
import json
import os
import re
import tempfile
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
from flask import Flask, render_template, request
from huggingface_hub import InferenceClient

load_dotenv()

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "students.json"

# Choose a model supported by a Hugging Face inference provider.
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Attendance marks are calculated out of 10.
MAX_MARKS = 10

VALID_STATUSES = {"P", "A", "NR"}

ACTIONS = {
    "get_attendance",
    "calculate_percentage",
    "calculate_marks",
    "remove_attendance",
    "mark_attendance",
    "class_average",
}


# ---------- JSON FILE FUNCTIONS ----------

def load_data():
    with DATA_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("students"), list):
        raise ValueError("Invalid students.json format.")

    rolls = set()

    for student in data["students"]:
        roll = str(student["roll_no"])

        if roll in rolls:
            raise ValueError(f"Duplicate roll number: {roll}")
        rolls.add(roll)

        attendance = student.get("attendance")

        if not isinstance(attendance, dict):
            raise ValueError(f"Invalid attendance for {roll}.")

        for day, status in attendance.items():
            if status not in VALID_STATUSES:
                raise ValueError(f"Invalid status for {roll}, {day}.")

    return data


def save_data(data):
    # Save a complete temporary file, then replace the original.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=BASE_DIR,
        suffix=".tmp",
        delete=False
    ) as file:
        temp_path = Path(file.name)
        json.dump(data, file, indent=2)
        file.write("\n")

    try:
        temp_path.replace(DATA_FILE)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def find_student(data, roll_no):
    for student in data["students"]:
        if str(student["roll_no"]) == str(roll_no):
            return student

    raise ValueError(f"Student roll number {roll_no} not found.")


# ---------- ATTENDANCE CALCULATIONS ----------

def attendance_stats(student):
    records = student["attendance"].values()

    recorded = [status for status in records
                if status in {"P", "A"}]

    total_days = len(recorded)
    present_days = sum(status == "P" for status in recorded)
    absent_days = total_days - present_days

    percentage = (
        present_days / total_days * 100
        if total_days else 0
    )

    return {
        "name": student["name"],
        "roll_no": str(student["roll_no"]),
        "total_days": total_days,
        "present_days": present_days,
        "absent_days": absent_days,
        "attendance_percentage": round(percentage, 2),
        "attendance_marks": round(
            percentage * MAX_MARKS / 100, 2
        ),
        "maximum_marks": MAX_MARKS,
    }


# ---------- PYTHON ATTENDANCE TOOLS ----------

def get_attendance(data, action):
    student = find_student(data, action["roll_no"])

    return {
        **attendance_stats(student),
        "daily_attendance": student["attendance"],
    }


def calculate_percentage(data, action):
    return attendance_stats(
        find_student(data, action["roll_no"])
    )


def calculate_marks(data, action):
    stats = attendance_stats(
        find_student(data, action["roll_no"])
    )

    return {
        "name": stats["name"],
        "roll_no": stats["roll_no"],
        "attendance_percentage": stats["attendance_percentage"],
        "attendance_marks": stats["attendance_marks"],
        "maximum_marks": stats["maximum_marks"],
    }


def remove_attendance(data, action):
    student = find_student(data, action["roll_no"])
    day = action["day"]

    if day not in student["attendance"]:
        raise ValueError(f"Unknown day: {day}")

    if student["attendance"][day] == "NR":
        raise ValueError("This attendance is already removed.")

    old_status = student["attendance"][day]
    student["attendance"][day] = "NR"

    save_data(data)

    return {
        "message": f"Attendance removed for {student['name']}, {day}.",
        "previous_status": old_status,
        "new_status": "NR",
        **attendance_stats(student),
    }


def mark_attendance(data, action):
    student = find_student(data, action["roll_no"])
    day = action["day"]
    status = action["status"]

    if day not in student["attendance"]:
        raise ValueError(f"Unknown day: {day}")

    if status not in {"P", "A"}:
        raise ValueError("Status must be P or A.")

    old_status = student["attendance"][day]
    student["attendance"][day] = status

    save_data(data)

    return {
        "message": f"{student['name']}: {day} marked {status}.",
        "previous_status": old_status,
        **attendance_stats(student),
    }


def class_average(data, action):
    results = [
        attendance_stats(student)
        for student in data["students"]
    ]

    percentages = [
        result["attendance_percentage"]
        for result in results
        if result["total_days"] > 0
    ]

    return {
        "student_count": len(results),
        "class_average_percentage": (
            round(mean(percentages), 2)
            if percentages else 0
        ),
        "students": results,
    }


TOOLS = {
    "get_attendance": get_attendance,
    "calculate_percentage": calculate_percentage,
    "calculate_marks": calculate_marks,
    "remove_attendance": remove_attendance,
    "mark_attendance": mark_attendance,
    "class_average": class_average,
}


# ---------- HUGGING FACE AI ----------

def get_ai_action(instruction, data):
    token = os.getenv("HF_TOKEN")

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it to your .env file."
        )

    client = InferenceClient(
        model=MODEL,
        token=token,
        provider="auto",
        timeout=60,
    )

    available_students = [
        {
            "name": student["name"],
            "roll_no": str(student["roll_no"]),
            "days": list(student["attendance"].keys()),
        }
        for student in data["students"]
    ]

    system_prompt = f"""
You convert attendance instructions into JSON commands.

Available actions:
- get_attendance: roll_no
- calculate_percentage: roll_no
- calculate_marks: roll_no
- remove_attendance: roll_no, day
- mark_attendance: roll_no, day, status
- class_average: no extra fields

Available students and days:
{json.dumps(available_students)}

Rules:
1. Return exactly one JSON object, no Markdown.
2. Use the exact roll number and day from the data.
3. Status must be P or A when marking attendance.
4. Never calculate percentages or marks yourself.
5. Never invent student records or attendance days.
6. If the request is ambiguous, return:
   {{"action":"invalid","reason":"Explain what is unclear"}}

Examples:
User: Calculate Rahul's percentage
JSON: {{"action":"calculate_percentage","roll_no":"101"}}

User: Remove Rahul's attendance on Day3
JSON: {{"action":"remove_attendance","roll_no":"101","day":"Day3"}}

User: Mark Amit absent on Day2
JSON: {{"action":"mark_attendance","roll_no":"103","day":"Day2","status":"A"}}

User: Calculate class average
JSON: {{"action":"class_average"}}
"""

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ],
        max_tokens=200,
        temperature=0,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The AI returned an empty response.")

    # Accept JSON with or without Markdown code fences.
    content = re.sub(
        r"^\s*```(?:json)?\s*", "", content.strip()
    )
    content = re.sub(r"\s*```\s*$", "", content)

    action = json.loads(content)

    if not isinstance(action, dict):
        raise ValueError("AI output must be a JSON object.")

    return action


# ---------- VALIDATION AND TOOL ROUTING ----------

def execute_action(data, action):
    if not isinstance(action, dict):
        raise ValueError("Invalid AI action.")

    name = action.get("action")

    if name not in ACTIONS:
        raise ValueError(
            action.get("reason", "Unsupported attendance action.")
        )

    required_fields = {
        "get_attendance": ["roll_no"],
        "calculate_percentage": ["roll_no"],
        "calculate_marks": ["roll_no"],
        "remove_attendance": ["roll_no", "day"],
        "mark_attendance": ["roll_no", "day", "status"],
        "class_average": [],
    }

    for field in required_fields[name]:
        if field not in action:
            raise ValueError(f"Missing field: {field}")

    # Validate the student, day and status before executing.
    if name != "class_average":
        student = find_student(data, action["roll_no"])

        if name in {"remove_attendance", "mark_attendance"}:
            if action["day"] not in student["attendance"]:
                raise ValueError("Unknown attendance day.")

        if name == "mark_attendance":
            if action["status"] not in {"P", "A"}:
                raise ValueError("Status must be P or A.")

    # Call the Python function selected by the AI.
    return TOOLS[name](data, action)


# ---------- WEB ROUTE ----------

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    instruction = ""

    try:
        data = load_data()
    except (OSError, ValueError, json.JSONDecodeError):
        return (
            "Could not load students.json. Check the file format.",
            500,
        )

    if request.method == "POST":
        instruction = request.form.get(
            "instruction", ""
        ).strip()

        if not instruction:
            error = "Please enter an instruction."
        elif len(instruction) > 500:
            error = "Instruction is too long."
        else:
            try:
                action = get_ai_action(instruction, data)

                # Reload current records before running the tool.
                data = load_data()
                result = execute_action(data, action)

            except (
                ValueError,
                KeyError,
                json.JSONDecodeError,
            ) as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("Attendance request failed")
                error = (
                    "The request failed. Check the Hugging Face "
                    "token, model availability and server terminal."
                )

    data = load_data()

    students = [
        {
            **attendance_stats(student),
            "attendance": student["attendance"],
        }
        for student in data["students"]
    ]

    percentages = [
        student["attendance_percentage"]
        for student in students
        if student["total_days"] > 0
    ]

    average = round(mean(percentages), 2) if percentages else 0

    return render_template(
        "index.html",
        students=students,
        average=average,
        instruction=instruction,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
