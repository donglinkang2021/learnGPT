# 如何添加一个新模型

写新模型的流程

1. 在 `src/models/` 中新建一个 `new_model.py`
2. 在 `src/models/__init__.py` 中注册模型
3. 在 `config/model/` 中新建一个 `new_model.yaml`
4. 在 `scripts/`中新建一个 `train_new_model.sh`

```sh
#!/bin/bash

# Prompt for the model name (file name)
read -p "Enter the model name (e.g., my_model): " model_name
model_name=$(echo "$model_name" | tr '[:upper:]' '[:lower:]') # lowercase it

# Prompt for the class name
read -p "Enter the class name (e.g., MyModel): " class_name

# 1. Create a new model file
echo "Creating src/models/${model_name}.py"
touch src/models/${model_name}.py

# 2. Register the model in __init__.py
echo "Registering the model in src/models/__init__.py"
echo "from .$model_name import $class_name" >> src/models/__init__.py

# 3. Create a new model config file
echo "Creating config/model/${model_name}.yaml"
touch config/model/${model_name}.yaml

# 4. Create a new training script
echo "Creating scripts/train_${model_name}.sh"
touch scripts/train_${model_name}.sh

echo "Done! Please modify the created files accordingly."
```
