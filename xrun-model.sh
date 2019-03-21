
pipenv run python xrun.py --description $1 --model xv1 --form t1 --label outcome --hyperparameters xhyperparameters.json --split e11fcc76-00b9-42ef-89a6-19ae7aef6f27
pipenv run python xrun.py --description $1 --model xv1 --form t2 --label outcome --hyperparameters xhyperparameters.json --split e11fcc76-00b9-42ef-89a6-19ae7aef6f27
pipenv run python xrun.py --description $1 --model xv1 --form features --label outcome --hyperparameters xhyperparameters.json --split e11fcc76-00b9-42ef-89a6-19ae7aef6f27