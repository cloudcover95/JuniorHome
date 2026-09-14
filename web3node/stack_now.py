import json
from missing_live import live as base
from palace_sis import palace
from q4k_shaped import report as q4
from trit3_layout import details
def live():
    row = base()
    row["trit3"] = details(); row["palace"] = palace(); row["q4k"] = q4()
    return row
if __name__ == "__main__":
    print(json.dumps(live(), indent=2, default=str))
