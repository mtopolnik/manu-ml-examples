import pandas as pd
import pickle
import json
import yaml
import util

inputs_folder = 'ai_inputs'
ai_folder = 'outputs1'
labels_dict = util.read_coll(ai_folder + '/labels_dict.json')
inv_labels_dict = {v: k for k, v in labels_dict.items()}
cols = util.read_coll(ai_folder + '/input_column_names.json')
occupation_stats = pd.read_csv(inputs_folder + '/occupation_stats.csv')
feature_selected_columns = util.read_coll(ai_folder + '/selected_features.txt')
inter_1 = pd.read_csv(ai_folder + '/education_mean.csv')
with open(inputs_folder + '/ai-config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.SafeLoader)
hours_info = util.read_coll(ai_folder + '/hours_info.json')
with open(ai_folder + '/onehot.model', 'rb') as f:
    encoder = pickle.load(f)
with open(ai_folder + '/rf.model', 'rb') as f:
    model = pickle.load(f)


def transform_list(input_json_list):
    row_dicts = [json.loads(input_json) for input_json in input_json_list]
    df = pd.DataFrame(row_dicts)
    df = df.merge(occupation_stats, how='inner', on='occupation')
    df = df[feature_selected_columns]
    df = df.merge(inter_1, on='education')
    df['age'] = config['age_multiplier'] * df['age']
    df['hours-per-week'] = df['hours-per-week'] - hours_info['min']
    df['hours-per-week'] = df['hours-per-week'] / (hours_info['max'] - hours_info['min'])
    x_test_encoded = encoder.transform(df)
    predicted_probabilities = model.predict_proba(x_test_encoded).transpose()[1]
    predicted_labels = model.predict(x_test_encoded).transpose()
    return [_to_json(proba, label)
        for proba, label in zip(predicted_probabilities, predicted_labels)]



def _to_json(predicted_probability, predicted_label):
    return json.dumps({
        'probability': float(predicted_probability),
        'label': inv_labels_dict[int(predicted_label)]
    })
