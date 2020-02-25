import pandas as pd


onehot_model_name = "onehot.model"
stage2_outputs = '../../stage2/outputs.json'

class Preprocess:
		
		def __init__(self, one_hot, target_column):
			self._one_hot = one_hot
			self._target_column = target_column
			
		
		def fit(self, df):
			if self._target_column in df:
				df = df.drop(self._target_column, axis = 1)
			self._categorical = [c for c in df.columns if str(df.dtypes[c]).lower().startswith('o')]
			self._remaining_columns = [c for c in df.columns if c not in self._categorical]
			categorical_df = df[self._categorical].fillna("NA")
			self._one_hot_model = self._one_hot.fit(categorical_df)
			#transformed_df = pd.DataFrame(self._one_hot_model.transform(categorical_df))
			#new_df = pd.concat([df[self._remaining_columns], transformed_df], axis = 1)
			#self._xgb_model = self._xgb.fit(new_df, labels)
		

		def transform(self, df):
			
			if self._target_column in df:
				df = df.drop(self._target_column, axis = 1)

			categorical_df = df[self._categorical].fillna("NA")
			transformed_df = pd.DataFrame(self._one_hot_model.transform(categorical_df))
			new_df = pd.concat([df[self._remaining_columns], transformed_df], axis = 1)
			return new_df
			#return self._xgb_model.predict_proba(new_df).transpose()[1]

