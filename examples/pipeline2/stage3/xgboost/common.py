import pandas as pd

stage2_outputs = '../../stage2/outputs.json'
ml_model_name = "ml.model"
target_column = 'default'


class MyXgboost:
		
		def __init__(self, one_hot, xgb_obj):
			self._one_hot = one_hot
			self._xgb = xgb_obj	 
		
		
		def fit(self, df):
			labels = df[target_column]
			df = df.drop(target_column, axis = 1)
			self._categorical = [c for c in df.columns if str(df.dtypes[c]).lower().startswith('o')]
			self._remaining_columns = [c for c in df.columns if c not in self._categorical]
			categorical_df = df[self._categorical].fillna("NA")
			self._one_hot_model = self._one_hot.fit(categorical_df)
			transformed_df = pd.DataFrame(self._one_hot_model.transform(categorical_df))
			new_df = pd.concat([df[self._remaining_columns], transformed_df], axis = 1)
			self._xgb_model = self._xgb.fit(new_df, labels)
		

		def transform(self, df):
			categorical_df = df[self._categorical].fillna("NA")
			transformed_df = pd.DataFrame(self._one_hot_model.transform(categorical_df))
			new_df = pd.concat([df[self._remaining_columns], transformed_df], axis = 1)
			return self._xgb_model.predict_proba(new_df).transpose()[1]

