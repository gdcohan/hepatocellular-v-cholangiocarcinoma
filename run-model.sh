pipenv run python run.py --description $1 --model v1 --form t1 --label outcome --hyperparameters hyperparameters.json --split e11fcc76-00b9-42ef-89a6-19ae7aef6f27
pipenv run python run.py --description $1 --model v1 --form t2 --label outcome --hyperparameters hyperparameters.json --split e11fcc76-00b9-42ef-89a6-19ae7aef6f27
pipenv run python run.py --description $1 --model v1 --form features --label outcome --hyperparameters hyperparameters.json --split e11fcc76-00b9-42ef-89a6-19ae7aef6f27
bash notify.sh "renal-mri all trials complete for $1"