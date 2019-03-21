from config import config
from db import db, Xresult, Calculatedresult
from sqlalchemy.sql import func
import json
import sys
import evaluate
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from numpy import array

# averaging calculations
def analyze_averages(parameters, description, model, input_form, split):

    for hyperparameter in parameters:
        json_hyperparameter = json.dumps(hyperparameter)

        # AVERAGE TESTING and HOLDOUT results across ALL trials
        averaged_test_f1 = db.session.query(func.avg(Xresult.holdout_f1_result)).filter(Xresult.description == description, Xresult.hyperparameters == json_hyperparameter, Xresult.input_form == input_form).scalar()
        averaged_test_accuracy = db.session.query(func.avg(Xresult.holdout_test_accuracy)).filter(Xresult.description == description, Xresult.hyperparameters == json_hyperparameter, Xresult.input_form == input_form).scalar()
        averaged_test_loss = db.session.query(func.avg(Xresult.holdout_test_loss)).filter(Xresult.description == description, Xresult.hyperparameters == json_hyperparameter, Xresult.input_form == input_form).scalar()
        averaged_holdout_f1 = db.session.query(func.avg(Xresult.holdout_f1_result)).filter(Xresult.description == description, Xresult.hyperparameters == json_hyperparameter, Xresult.input_form == input_form).scalar()
        averaged_holdout_accuracy = db.session.query(func.avg(Xresult.holdout_test_accuracy)).filter(Xresult.description == description, Xresult.hyperparameters == json_hyperparameter, Xresult.input_form == input_form).scalar()
        averaged_holdout_loss = db.session.query(func.avg(Xresult.holdout_test_loss)).filter(Xresult.description == description, Xresult.hyperparameters == json_hyperparameter, Xresult.input_form == input_form).scalar()

        # Ensembling the k-fold models by minimum loss
        min_loss_f1, min_loss_accuracy, min_loss_test_acc, min_loss_test_loss = ensemble_folds(json_hyperparameter, description, "min", input_form)
        # Ensembling the k-fold models by maximum accuracy
        max_acc_f1, max_acc_accuracy, max_acc_test_acc, max_acc_test_loss = ensemble_folds(json_hyperparameter, description, "min", input_form)

        # save it all
        result = Calculatedresult(
            str(split),
            model,
            min_loss_f1,
            min_loss_accuracy,
            min_loss_test_acc,
            min_loss_test_loss,
            max_acc_f1,
            max_acc_accuracy,
            max_acc_test_acc,
            max_acc_test_loss,
            averaged_test_f1,
            averaged_test_accuracy,
            averaged_test_loss,
            averaged_holdout_f1,
            averaged_holdout_accuracy,
            averaged_holdout_loss,
            hyperparameter,
            description,
            input_form
        )
        db.session.add(result)
        db.session.commit()
    return


# for each n-fold go through each trial to pick model with LEAST TESTING LOSS for a total of n-final models per hyperparameter
# mode is min for now
def ensemble_folds(json_hyperparameter, description, mode, input_form):
    least_loss_list = list()
    # get the trials with the least loss for each fold

    if mode == "min":
        for x in range(config.NUMBER_OF_FOLDS - 1):
            subquery = db.session.query(func.min(Xresult.test_loss)).filter(Xresult.description == description,
                                                                            Xresult.hyperparameters == json_hyperparameter,
                                                                            Xresult.input_form == input_form,
                                                                            Xresult.fold == x + 1)
            model_with_least_loss = db.session.query(Xresult).filter(Xresult.description == description,
                                                                     Xresult.hyperparameters == json_hyperparameter,
                                                                     Xresult.input_form == input_form,
                                                                     Xresult.fold == x + 1,
                                                                     Xresult.test_loss == subquery).first()
            least_loss_list.append(model_with_least_loss)

    if mode == "max":
        for x in range(config.NUMBER_OF_FOLDS - 1):
            subquery = db.session.query(func.max(Xresult.test_accuracy)).filter(Xresult.description == description,
                                                                                Xresult.hyperparameters == json_hyperparameter,
                                                                                Xresult.input_form == input_form,
                                                                                Xresult.fold == x + 1)
            model_with_most_accuracy = db.session.query(Xresult).filter(Xresult.description == description,
                                                                        Xresult.hyperparameters == json_hyperparameter,
                                                                        Xresult.input_form == input_form,
                                                                        Xresult.fold == x + 1,
                                                                        Xresult.test_loss == subquery).first()
            least_loss_list.append(model_with_most_accuracy)

    # average the probabilities of the n models for the n folds selected
    flag = 0
    holdout_labels = list()
    holdout_probs = list()
    test_loss = 0
    test_acc = 0

    for y in least_loss_list:
        fold_holdout_probs = y.get_holdout_probabilities()
        check_holdout_labels = y.get_holdout_labels()
        test_loss = test_loss + y.test_loss
        test_acc = test_acc + y.test_accuracy

        # test to make sure the holdout label list is the same across all trials
        if flag == 0:
            holdout_labels = check_holdout_labels
            holdout_probs = fold_holdout_probs
            flag = 1
        elif holdout_labels != check_holdout_labels:
            sys.stderr.write("Holdout labels for this run: " + description + " did not match!")
            return
        else:
            holdout_probs = [x + y for x, y in zip(holdout_probs, fold_holdout_probs)]

    avg_holdout_probs = [i / len(holdout_probs) for i in holdout_probs]

    # convert averaged probabilities to binary
    array_avg_holdout_probs = array(avg_holdout_probs)
    holdout_binary_predictions = list(evaluate.transform_binary_predictions(array_avg_holdout_probs))

    # using the probabilities, test it against the holdout set
    holdout_f1_result = f1_score(holdout_labels, holdout_binary_predictions)
    holdout_accuracy_result = accuracy_score(holdout_labels, holdout_binary_predictions)

    # also return the averaged test accuracy and loss of the selected models
    test_loss = test_loss / len(least_loss_list)
    test_acc = test_acc / len(least_loss_list)

    return holdout_f1_result, holdout_accuracy_result, test_acc, test_loss
