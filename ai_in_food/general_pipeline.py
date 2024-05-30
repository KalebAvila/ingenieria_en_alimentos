import subprocess

command_list = [
    'python src/data/get_data.py',
    'python src/features/build_features.py',
    'python src/models/prepare_data.py',
    'python src/models/train_test_split.py',
    'python src/models/train_model.py',
    'python src/visualization/evaluate_model.py',
    'python src/demo/app.py'
]

for command in command_list:
    subprocess.run(command.split(' '))
