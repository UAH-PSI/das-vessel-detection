def load_model():
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=10,
        n_estimators=500,
        random_state=42,
    )
