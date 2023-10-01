# sussex-huawei-locomotion-challenge2023
This is the repository of the TDU_BSA team that participated in the sussex-huawei-locomotion-challenge2023.

This section summarizes the TDU_BSA team's solution approach. (Incomplete)

## Preparation for operation
Download the data to be used from [here](http://www.shl-dataset.org/activity-recognition-challenge-2023/)http://www.shl-dataset.org/activity-recognition-challenge-2023/ and place it in `/data/raw`.
```
├ data
  ├ raw
    ├ SHL-2023-Train
    ├ SHL-2023-Validate
    ├ SHL-2023-Test
```

# How to run
Execute main.py in the directory under script.

Execute in the following order. May not work due to incomplete

1. script/Preprocessing/main.py
2. script/LSTM_label_estimation/main.py
3. script/XGBoost/main.py
4. script/EnsembleLearning/main.py
5. script/PostProcessing/main.py
