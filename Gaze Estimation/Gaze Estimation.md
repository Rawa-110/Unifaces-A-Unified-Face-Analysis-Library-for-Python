Gaze Estimation

Gaze estimation is the task of predicting the direction in which a person is looking. The model estimates two angles:

Pitch — the vertical direction of the gaze (up/down).
Yaw — the horizontal direction of the gaze (left/right).

The table compares different backbones in terms of model size and gaze-estimation accuracy on the Gaze360 test set.

<img width="1864" height="722" alt="image" src="https://github.com/user-attachments/assets/cd588f00-239c-484e-99fc-4ba2be860f4c" />


| Backbone     | Model Size |      MAE ↓ | Interpretation                          |
| ------------ | ---------: | ---------: | --------------------------------------- |
| ResNet18     |      43 MB |     12.84° | Moderate accuracy                       |
| **ResNet34** |      82 MB | **11.33°** | **Best accuracy**                       |
| ResNet50     |      91 MB |     11.34° | Almost identical to ResNet34            |
| MobileNetV2  |     9.6 MB |     13.07° | Lightweight but less accurate           |
| MobileOne-S0 | **4.8 MB** |     12.58° | Smallest model with reasonable accuracy |



What does MAE mean?

MAE (Mean Absolute Error) measures how far the predicted gaze angles are from the actual angles.

For example, if the actual gaze direction is:

Pitch = 10°
Yaw   = 20°

and the model predicts:

Pitch = 12°
Yaw   = 17°

the errors are approximately:

Pitch error = |12 - 10| = 2°
Yaw error   = |17 - 20| = 3°

A lower MAE is better, because it means the predicted gaze direction is closer to the ground truth.

Main takeaway
ResNet34 gives the best MAE: 11.33°.
ResNet50 performs essentially the same: 11.34°, despite being larger.
MobileNetV2 is much smaller (9.6 MB) but sacrifices some accuracy.
MobileOne-S0 is the smallest model (4.8 MB) while still achieving a relatively good 12.58° MAE.

So, if accuracy is the priority → ResNet34.
If model size and efficiency are the priority → MobileOne-S0.
