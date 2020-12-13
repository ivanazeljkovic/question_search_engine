# Question Search Engine

## Project Setup
### Local setup
Project requires `git`, `python >= 3.6` with `pip` and `virtualenv` (optionally `virtualenvwrapper`).
1. Install Python 3.6
2. Install pip
3. Install virtualenv (virtualenvwrapper optionally)
4. System libraries (as support for Python libraries)

Clone repository:

```bash
git clone https://github.com/ivanazeljkovic/question_search_engine.git
cd question_search_engine/
```

Create virtual environment with:

```bash
virtualenv -p python3.6 venv
```

or if you are using `virtualenvwrapper` instead of `virtualenv`:

```bash
mkvirtualenv -p python3.6 venv
```

### Requirements

Install requirements with activated virtual environment:

```bash
pip install -r requirements.txt
```

## Project Running
### Dataset 
Navigate to **/data/raw** directory and store **questions.json** file with questions corpus.
The structure of corpus file should be the same as shown in the example below:
```
{"id": 1, "question": "what is TF-IDF?", "tags": "<nlp>"}
{"id": 2, "question": "should I ignore poentry.lock?", "tags": "<python>"}
{"id": 3, "question": "How to use pytest?", "tags": "<python><pytest>"}
```

### Running
From root directory run:
```
python run.py
```
Wait for processes of corpus loading and fitting into TF-IDF vectorizer to be done. 
When an interactive prompt is open, input a question of interest:
```
>>> Error handling in Java?
```
The structure of output should be the same as shown in the example below:
```
0.8318 43953635 How do I use Error handling in Java
0.7683 38835571 Error Handling in Swift 3
0.6029 47684377 Java BufferedReader error
0.5649 38936305 If block error handling in bash
0.5519 52513360 java ATM program simulation with exception handling - no error neither full output.
```

### Testing
#### 1. Particular group of Unit tests
From root directory run command for running a particular group of Unit tests:
```
python -m tests.test_preprocessor
python -m tests.test_tf_idf_vectorizer
python -m tests.test_utils
```
#### 2. All Unit tests
From root directory run commands for running shell scipt:
```
chmod +x tests/run_all_tests.sh
tests/run_all_tests.sh
```
