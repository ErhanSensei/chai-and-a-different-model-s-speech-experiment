from transformers import GPT2LMHeadModel, GPT2Tokenizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import time
import torch
import sys

# Set console output to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", message=".*Flash Attention.*")

# Hugging Face GPT-2 model and tokenizer installation
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# If you have a GPU, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Options for Chrome
chrome_options = Options()
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-popup-blocking")
chrome_options.add_argument("--disable-blink-features=AutomationControlled")
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
chrome_options.add_experimental_option("useAutomationExtension", False)

# Initialize ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# Generate a question based answer from GPT-2
def generate_question_gpt2(last_response, max_length=50):
    prompt = f"Given the following response, generate a relevant follow-up question:\nResponse: {last_response}\nQuestion:"

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs,
        max_length=len(inputs[0]) + max_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Temperature parameter to add randomness
        top_p=0.9  # Nucleus sampling to consider various possibilities
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Just take the newly created part
    question = generated_text[len(prompt):].strip()
    return question

# Open the web page
driver.get("https://character.ai/login")

print("Please sign in with your Google account or E-mail. Waiting 40 seconds...")
time.sleep(40)

print("Trying to redirect to character page...")
time.sleep(5)

driver.get("Add the link of the bot you want to interact with on character.ai here")

time.sleep(5)
if driver.current_url != "Add the link of the bot you want to interact with on character.ai here":
    print("Returned to home page. Redirection failed.")
else:
    print("Successfully redirected to character page.")

# Speech loop
previous_question = "Type the question you want GPT-2 to ask here."

# Loop
for i in range(50):
    try:
        # Find the user input field
        chat_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[@id='chat-body']/div[2]/div/div/div/div[1]/textarea"))
        )

        # Submit first question or continue from previous question
        chat_input.send_keys(previous_question)
        chat_input.send_keys(Keys.RETURN)

        # Wait for the response
        response_element = WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//*[@data-testid='completed-message']/div/div/p"))
        )
        response = response_element.text
        print(f"Character AI Response ({i+1}):", response)

        # Create a new question
        new_question = generate_question_gpt2(response)
        if new_question:
            print(f"GPT-2 New Question ({i+1}):", new_question)
            previous_question = new_question  # Save question for next loop
        else:
            print("Could not create a new question. Ending loop.")
            break

        time.sleep(2)

    except Exception as e:
        print(f"Error ({i+1}): {e}")
        break
