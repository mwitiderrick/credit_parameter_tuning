"""Credit Approval

This file demonstrates how we can develop and train our credit_approval_model_bayesian_search by using the
`features` we've developed earlier. Every ML credit_approval_model_bayesian_search project
should have a definition file like this one.

"""
from typing import Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import F1
from sklearn.ensemble import RandomForestClassifier
from layer import Featureset, Train


def train_model(train: Train, pf: Featureset("credit_approval_features")) -> Any:
    """Model train function

    This function is a reserved function and will be called by Layer
    when we want this credit_approval_model_bayesian_search to be trained along with the parameters.
    Just like the `features` featureset, you can add more
    parameters to this method to request artifacts (datasets,
    featuresets or models) from Layer.

    Args:
        train (layer.Train): Represents the current train of the credit_approval_model_bayesian_search, passed by
            Layer when the training of the credit_approval_model_bayesian_search starts.
        pf (spark.DataFrame): Layer will return all features inside the
            `features` featureset as a spark.DataFrame automatically
            joining them by primary keys, described in the dataset.yml

    Returns:
       credit_approval_model_bayesian_search: Trained credit_approval_model_bayesian_search object

    """
    # We create the training and label data
    train_df = pf.to_pandas()
    X = train_df.drop(["ID", "approved"], axis=1)
    Y = train_df["approved"]

    random_state = 25
    test_size = 0.3
    train.log_parameter("random_state", random_state)
    train.log_parameter("test_size", test_size)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                        random_state=random_state)

    # Here we register input & output of the train. Layer will use
    # this registers to extract the signature of the credit_approval_model_bayesian_search and calculate
    # the drift
    train.register_input(X_train)
    train.register_output(y_train)

    n_estimators = train.get_parameter("n_estimators")
    max_depth = train.get_parameters("max_depth")
    bootstrap = train.get_parameters("bootstrap")

    random_forest = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        bootstrap=bootstrap
    )

    random_forest.fit(X_train, y_train)

    # making predictions
    y_preds = random_forest.predict(X_test)
    f1 = f1_score(y_test, y_preds)
    train.log_metric("f1", f1)

    # We return the credit_approval_model_bayesian_search
    return random_forest