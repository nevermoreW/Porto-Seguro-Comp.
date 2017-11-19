from xgboost import XGBClassifier




def naive_optimal_xgb_model():
    model=XGBClassifier(n_estimators=1,
                  max_depth=4,
                  objective="binary:logistic",
                  learning_rate=0.07,
                  subsample=0.8,
                  min_child_weight=6,
                  colsample_bytree=0.8,
                  scale_pos_weight=1.6,
                  gamma=10,
                  reg_alpha=8,
                  reg_lambda=1.3)
    return model
