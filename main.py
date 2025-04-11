"""
SciQuizPro - AI-Powered Science Quiz Generator
Using a locally-hosted fine-tuned Mistral 7B model with Ollama

This app generates educational science quizzes across various topics
and difficulty levels using a custom-trained LLM optimized for
educational content.
"""

from flask import Flask, render_template, request, jsonify
import json
import os
import requests
import time
import re
import logging

# Import model configuration
from config.model_config import (
    OLLAMA_API_URL,
    MODEL_NAME,
    TEMPERATURE,
    TOP_P,
    MAX_TOKENS,
    STOP_SEQUENCES,
    SCIENCE_CATEGORIES
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


# Routes
@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


@app.route('/generator')
def generator():
    """Render the quiz generator page"""
    return render_template('generator.html')


@app.route('/generate_quiz', methods=['POST'])
def generate_quiz():
    """
    Generate a quiz based on the provided parameters.

    Accepts POST requests with JSON data containing:
    - topic: The general science category
    - subtopic: The specific topic within the category
    - numQuestions: Number of questions to generate
    - gradeLevel: The educational level (elementary, middle, high school, college)
    - temperature: Controls randomness in generation (optional)
    - top_p: Controls diversity via nucleus sampling (optional)
    - max_tokens: Maximum response length (optional)

    Returns a JSON object with the generated quiz or error message.
    """
    # Extract data from request
    data = request.json
    topic = data.get('topic')
    subtopic = data.get('subtopic')
    num_questions = int(data.get('numQuestions', 5))
    grade_level = data.get('gradeLevel', 'middle school')

    # Model parameters (use defaults if not provided)
    temperature = float(data.get('temperature', TEMPERATURE))
    top_p = float(data.get('top_p', TOP_P))
    max_tokens = int(data.get('max_tokens', MAX_TOKENS))

    logger.info(f"Generating quiz: {subtopic} ({topic}) - {num_questions} questions - {grade_level} level")
    logger.info(f"Model parameters: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}")

    # Use the Ollama API to generate the quiz
    try:
        # Create the prompt for the model
        prompt = create_prompt(topic, subtopic, num_questions, grade_level)

        # Call the Ollama API with retries
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                quiz_data = call_ollama_api(
                    prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                # Validate response
                if len(quiz_data) != num_questions:
                    logger.warning(
                        f"Received {len(quiz_data)} questions instead of {num_questions}. Truncating or retrying.")
                    if len(quiz_data) > num_questions:
                        # If we got too many questions, just truncate
                        quiz_data = quiz_data[:num_questions]
                    else:
                        # If we got too few questions, retry
                        retry_count += 1
                        continue

                # Validate each question structure
                valid_questions = True
                for i, q in enumerate(quiz_data):
                    if not all(key in q for key in ['question', 'options', 'correctAnswer']):
                        logger.warning(f"Question {i + 1} is missing required fields")
                        valid_questions = False
                        break

                    # Ensure options is a list with exactly 4 items
                    if not isinstance(q['options'], list) or len(q['options']) != 4:
                        logger.warning(f"Question {i + 1} does not have exactly 4 options: {q['options']}")
                        valid_questions = False
                        break

                    # Ensure correctAnswer is in options
                    if q['correctAnswer'] not in q['options']:
                        logger.warning(f"Question {i + 1} has incorrect answer not in options: {q['correctAnswer']}")
                        valid_questions = False
                        break

                if not valid_questions:
                    retry_count += 1
                    continue

                # If we got here, validation passed
                break

            except Exception as e:
                logger.error(f"Attempt {retry_count + 1} failed: {e}")
                retry_count += 1
                time.sleep(2)  # Wait before retrying

        # If we exhausted retries, raise the last error
        if retry_count >= max_retries:
            raise Exception(
                f"Failed to generate valid quiz after {max_retries} attempts. Please check if Ollama is running properly.")

        return jsonify({'success': True, 'quiz': quiz_data})

    except Exception as e:
        logger.error(f"Error generating quiz with Ollama: {e}")
        error_message = f"Failed to generate quiz. Error: {str(e)}. Please ensure Ollama is running with your fine-tuned model."
        return jsonify({'success': False, 'error': error_message})


def create_prompt(topic, subtopic, num_questions, grade_level):
    """
    Create a prompt for the fine-tuned Mistral 7B model.

    Parameters:
    - topic: The general science category
    - subtopic: The specific topic within the category
    - num_questions: Number of questions to generate
    - grade_level: The educational level

    Returns a string prompt optimized for the model.
    """
    return f"""You are a dedicated educational content creator. Create a science quiz on the topic of "{subtopic}" within the {topic} category for {grade_level} students.

IMPORTANT INSTRUCTIONS:
1. Create EXACTLY {num_questions} multiple-choice questions - no more, no less.
2. Each question must have EXACTLY 4 options labeled A, B, C, and D.
3. For each question, clearly indicate the correct answer.
4. All questions should be factual, educational, and appropriate for {grade_level} students.
5. Do not include any markdown formatting, explanations, or introductory text.
6. The output must be ONLY valid JSON in the format specified below.

Return ONLY this JSON array with {num_questions} question objects:
[
  {{
    "question": "Question text here?",
    "options": ["Option A text", "Option B text", "Option C text", "Option D text"],
    "correctAnswer": "The exact text of the correct option"
  }},
  ...
]

Ensure:
- The 'correctAnswer' value exactly matches one of the strings in the 'options' array.
- All content is scientifically accurate and educational.
- Your response contains EXACTLY {num_questions} questions.
- Your response is ONLY the JSON array, with no additional text before or after.
"""


def call_ollama_api(prompt, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS):
    """
    Call the Ollama API to generate quiz questions using the fine-tuned Mistral 7B model.

    Parameters:
    - prompt: The text prompt for the model
    - temperature: Controls randomness (0.0 to 1.0)
    - top_p: Controls diversity via nucleus sampling
    - max_tokens: Maximum number of tokens to generate

    Returns the parsed JSON quiz data.
    """
    # Prepare request for Ollama API
    request_data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "stop": STOP_SEQUENCES
        }
    }

    try:
        logger.info(f"Calling Ollama API with model: {MODEL_NAME}")
        response = requests.post(OLLAMA_API_URL, json=request_data, timeout=120)
        response.raise_for_status()

        result = response.json()

        # Extract the response from Ollama
        if 'response' in result:
            content = result['response'].strip()
            logger.info(f"Received response from Ollama (first 100 chars): {content[:100]}...")

            # Clean up the content for JSON parsing
            cleaned_content = content

            # Remove markdown code block markers if present
            cleaned_content = re.sub(r'^```json\s*', '', cleaned_content)
            cleaned_content = re.sub(r'\s*```$', '', cleaned_content)
            cleaned_content = re.sub(r'^```\s*', '', cleaned_content)

            # Try direct JSON parsing with the cleaned content
            try:
                quiz_json = json.loads(cleaned_content)
                return quiz_json
            except json.JSONDecodeError as json_err:
                logger.error(f"JSONDecodeError with cleaned content: {json_err}")

                # Fallback strategy 1: Try to extract JSON array
                array_match = re.search(r'\[\s*{.+}\s*\]', cleaned_content, re.DOTALL)
                if array_match:
                    try:
                        extracted_array = array_match.group(0)
                        logger.info(f"Attempting to parse extracted array")
                        return json.loads(extracted_array)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse extracted array")

                # Fallback strategy 2: Try to extract individual JSON objects
                objects = re.findall(r'{[^{}]*"question"[^{}]*"options"[^{}]*"correctAnswer"[^{}]*}', cleaned_content)
                if objects:
                    try:
                        reconstructed = "[" + ",".join(objects) + "]"
                        return json.loads(reconstructed)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse reconstructed array from objects")

                # If all parsing attempts failed
                raise Exception(f"Could not parse model response into valid JSON. Response: {content[:300]}...")
        else:
            raise Exception("Ollama API response does not contain expected 'response' field.")

    except requests.exceptions.RequestException as req_err:
        logger.error(f"RequestException: Could not connect to Ollama API: {req_err}")
        raise Exception(f"Could not connect to Ollama API. Please check if Ollama is running locally.") from req_err
    except Exception as e:
        logger.error(f"Unexpected error in call_ollama_api: {e}")
        raise


if __name__ == '__main__':
    logger.info(f"Starting SciQuizPro Flask application...")
    logger.info(f"Using Mistral 7B fine-tuned model: {MODEL_NAME}")
    logger.info(f"Ollama API URL: {OLLAMA_API_URL}")
    logger.info(f"Visit http://127.0.0.1:5000 to access the application")
    app.run(debug=True)