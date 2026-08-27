Attributes
FaceAttribNet reads five independent binary states per face: left and right eye openness, eyeglasses, face mask, and sunglasses


<img width="1762" height="574" alt="image" src="https://github.com/user-attachments/assets/2c989ade-1bba-4355-900c-07ece10c1fc5" />






| Attribute              | Possible states | Meaning                                          |
| ---------------------- | --------------- | ------------------------------------------------ |
| **Left Eye Openness**  | Open / Closed   | Whether the left eye is open                     |
| **Right Eye Openness** | Open / Closed   | Whether the right eye is open                    |
| **Eyeglasses**         | Yes / No        | Whether the person is wearing regular eyeglasses |
| **Face Mask**          | Yes / No        | Whether the person is wearing a face mask        |
| **Sunglasses**         | Yes / No        | Whether the person is wearing sunglasses         |


What does “five independent binary states” mean?

Each attribute is classified independently as either 0 or 1:

0 → attribute is absent / condition is false
1 → attribute is present / condition is true

For example:

Left Eye Open   = 1
Right Eye Open  = 1
Eyeglasses      = 1
Face Mask       = 0
Sunglasses      = 0

This means:

Both eyes are open, the person is wearing eyeglasses, and they are not wearing a mask or sunglasses.

The five outputs can therefore be represented as a vector:

[LeftEye, RightEye, Eyeglasses, FaceMask, Sunglasses]

For example:

[1, 1, 0, 1, 0]

means:

Left eye: open
Right eye: open
Eyeglasses: no
Face mask: yes
Sunglasses: no

The key idea is that the model predicts five separate attributes rather than assigning the entire face to one single class.

