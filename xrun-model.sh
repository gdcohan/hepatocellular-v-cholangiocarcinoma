
pipenv run python xrun.py --description $1 --model v1 --form t1 --label outcome --hyperparameters xhyperparameters.json --split 42
pipenv run python xrun.py --description $1 --model v1 --form t2 --label outcome --hyperparameters xhyperparameters.json --split 42
pipenv run python xrun.py --description $1 --model v1 --form features --label outcome --hyperparameters xhyperparameters.json --split 42