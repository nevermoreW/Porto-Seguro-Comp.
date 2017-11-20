from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier




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

def naive_optimal_rf_model():
    model=RandomForestClassifier(n_estimators=1000, n_jobs=-1, class_weight="balanced", min_samples_leaf=25, min_samples_split=25)
    return model
