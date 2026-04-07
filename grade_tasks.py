import json

from graders import grade_all_tasks


if __name__ == "__main__":
    result = grade_all_tasks()
    print(json.dumps(result, indent=2))
