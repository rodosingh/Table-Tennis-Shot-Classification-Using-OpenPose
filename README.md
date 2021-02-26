# Table-Tennis-Shot-Classification-Using-OpenPose
Pose Classification using OpenPose from OpenCV library.
<h3>Objective</h3>

To classify whether the shot played in the given video is ***backhand*** or ***forehand***.

- Clone [this](https://github.com/rodosingh/Table-Tennis-Shot-Classification-Using-OpenPose) github repo to perform Table-Tennis shot classification on your own. 
- Ensure that you have met all the requirements mentioned at the end of this **Readme** file.
- Open your terminal in the cloned folder and run `python3 Pose_Estimation.py`
- You will get your required output!!!

User must provide inputs as follows:
```python
video_path = # relative path or absolute path to your final data
video_side = # Among two, player of which side is your concern (say left/right).
hand_type = # which hand is dominant while taking shot (say lefty/righty).
```
You will get output as: `Video of Backhand shot` or `Video of Forehand shot`

### Methodology

1. Take 20 frames of video and crop it to just focus on player of one side.
2. Using `OpenPose` pose estimation technique in `OpenCV` library, I estimated the keypoints of body (as shown below) using heatMap technique for each frame and collected them for all frames which serves as the data of one sample (here it is video).
```python 
BODY_PARTS_C = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4, "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "LHip": 11}
```
3. I have created four functions (models) where each trains on extracted data from videos of particular criteria. Since we have `video_side = left or right` (2 options) and `hand_type = lefty or righty` (2 options), in total there are 4 possibilities and hence 4 models, each dealing with a particular scenario.
4. Each model consists of 6 robust machine learning classifiers (`XGBoost`, `LogisticRegression`, `MLPClassifier`, `ExtraTreesClassifier`, `RandomForestClassifier`, and `KNeighborsClassifer`) which predict some values in between 0 and 1 (a probabilistic prediction) for a given video sample. Predictions of all 6 models are stacked and again feed into a meta-learner (which is either `XGBClassifier` or `LogisticRegression`) that learns how to predict output (basically a stacking ensemble).
5. Now the output from this meta-learner is final output that says whether **the shot is backhand or forehand**.

- [Link to Complete Jupyter Notebook](https://github.com/rodosingh/Table-Tennis-Shot-Classification-Using-OpenPose/blob/main/Pose_Estimation.ipynb)
- [Data - Video Files](https://github.com/rodosingh/Table-Tennis-Shot-Classification-Using-OpenPose/tree/main/video_data)
- [Extracted Video Data in `.npz` compressed format](https://github.com/rodosingh/Table-Tennis-Shot-Classification-Using-OpenPose/tree/main/engineered_data)
- [Pretrained Models](https://github.com/rodosingh/Table-Tennis-Shot-Classification-Using-OpenPose/tree/main/best_models)
- [Meta Models](https://github.com/rodosingh/Table-Tennis-Shot-Classification-Using-OpenPose/tree/main/meta_models)

### Requirements
```
1. Open CV
2. Scikit-Learn
3. Numpy
4. Pandas
5. Matplotlib
6. Pickle
```
