# app.py
from core import ask

def main():
    print("\n[READY] Ask me anything. Type 'exit' to quit.\n")
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            break
        print("Bot:", ask(question), "\n")

if __name__ == "__main__":
    main()
