from openai import OpenAI
import pytest
import os
from fastapi.testclient import TestClient
import json
from dotenv import load_dotenv
from app.main import app

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def ans_matching(str1, str2) -> bool:
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an AI that compares two strings and determines if they are logically equivalent. The output should be a boolean - true or false"
            },
            {
                "role": "user",
                "content": f"Are the following two strings logically equivalent?\n\nString 1: {str1}\nString 2: {str2}"
            }
        ],
        model="gpt-4",
    )
    result = response.choices[0].message.content
    return result


@pytest.fixture
def authorized_client():
    client = TestClient(app)
    return client

def test_server_runs(authorized_client):
    res = authorized_client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "Welcome to Bolt"}
    
def test_irrelevant_questions(authorized_client):
    user_question = {"question": "How to dance on the beach?"}
    res = authorized_client.post("/ask/", json=user_question)
    response_data = res.json()
    expected_answer = "I can only help you with academic advising"
    assert res.status_code == 200
    assert ans_matching(response_data["question"], expected_answer)


def test_credits_inquiry(authorized_client):
    user_question = {
        "question": "I currently have 72 credits. How many more for fourth year standing?"}
    res = authorized_client.post("/ask/", json=user_question)
    response_data = res.json()
    expected_answer = "You would need 6 more credits to get fourth year standing"
    assert res.status_code == 200
    assert ans_matching(response_data["question"], expected_answer)


def test_transfer_credits_inquiry(authorized_client):
    user_question = {
        "question": "I completed IB Visual Arts SL with a grade of 4. Will I get transfer credits? and if yes, then for which equivalent course at UBC?"}
    res = authorized_client.post("/ask/", json=user_question)
    response_data = res.json()
    expected_answer = "Yes, you would recieve 3 credits as transfer credits. The equivalent course at UBC is VISA102"
    assert res.status_code == 200
    assert ans_matching(response_data["question"], expected_answer)


def test_graduation_inquiry(authorized_client):
    user_question = {
        "question": "I want to ship my parchement to Spain. What would be the estimated cost and how many days would it take?"}
    res = authorized_client.post("/ask/", json=user_question)
    response_data = res.json()
    expected_answer = "The estimated cost would be $43.60 and it would three to five business days"
    assert res.status_code == 200
    assert ans_matching(response_data["question"], expected_answer)


def test_major_minor_inquiry(authorized_client):
    user_question = {
        "question": "Can students self-declare any major within the BSc program?"}
    res = authorized_client.post("/ask/", json=user_question)
    response_data = res.json()
    expected_answer = "Yes, you can self-declare any major within the BSc program"
    assert res.status_code == 200
    assert ans_matching(response_data["question"], expected_answer)


def test_message_history(authorized_client):
    user_question1 = {"question": "Hi there, my name is John"}
    res1 = authorized_client.post("/ask/", json=user_question1)
    user_question2 = {"question": "What is my name?"}
    res2 = authorized_client.post("/ask/", json=user_question2)
    response_data = res2.json()
    expected_answer = "Your name is John"
    assert ans_matching(response_data["question"], expected_answer)
