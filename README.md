```
# ReviewRanker: A Semi-Supervised Learning Based Approach 
# for Code Review Quality Estimation
```

- Required python version >= 3.8.10
- Create virtual environment and activate venv in project
```shell script
$ python -m venv env
$ source venv/bin/activate
```

#### Install project dependencies 
- Install libraries using requirements.txt
```shell script
$ pip install -r requirements.txt
```

#### Run 
- You need the full path of your code review dataset
- It will generate an Excel with the confidence score of each review
```shell script
$ python code/main.py /path/code_review.csv
```

