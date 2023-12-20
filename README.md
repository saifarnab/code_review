```
Title:
ReviewRanker: A Semi-Supervised Learning Based Approach 
for Code Review Quality Estimation
```

```
Authors:
Saifullah Mahbub[1] saifornab@gmail.com
Md. Easin Arafat[1,2]
Chowdhury Rafeed Rahman[1,4]
Zannatul Ferdows[1]
Masum Hasan[3,5]

1 Department of Computer Science and Engineering, United International University, United City, Madani Avenue, Dhaka, 1212, Bangladesh
2 Data Science and Engineering Department, Faculty of Informatics, Eötvös Loránd University, Pázmány Péter str. 1/A, Budapest, 1117, Hungary
3 Department of Computer Science and Engineering, Bangladesh University of Engineering and Technology, Dhaka, 1000, Bangladesh
4 School of Computing, National University of Singapore, Singapore
5 Department of Computer Science, University of Rochester, Rochester, NY, United States
```

# How to run:  
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

