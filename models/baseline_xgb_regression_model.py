def load_model(seed=42):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=10,
        n_estimators=500,
        random_state=seed,
    )
