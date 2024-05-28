import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pickle
def decision_tree_model(df: pd.DataFrame(), input_columns: list, predict_column: str, ticker_name: str):
    X = df[input_columns]
    y = df[predict_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model = DecisionTreeRegressor(max_depth=10,criterion="squared_error",random_state=70)
    model.fit(X_train, y_train)

    with open(f"source_files/{ticker_name}_random_forest.pkl", "wb") as f:
        pickle.dump(model, f)