from golden_dataset_mcp.models import EvaluateInput
from golden_dataset_mcp.server import evaluate_answers

print("Starting evaluate...")

result = evaluate_answers(
    EvaluateInput(
        dataset_path=r"C:\Users\Nipun\Downloads\test-golden-ds",
        actual_answers=["Heathrow Terminal 5", "Airbus A380"],
    )
)

print("Done!")
print(result)
