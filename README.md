# PANDA

|Date|Model| Augmentation |Image Size| Batch Size |Validation | LB |
|----|-----|------------|------------|------------|-----------|----|
|April-27-16:28:23|EfficientNet-B0|HorizontalFlip,VerticalFlip|384 * 384|16|0.7287| 0.64
|April-27-23:24:32|EfficientNet-B1|HorizontalFlip,VerticalFlip|384 * 384|16|0.7588| 0.62
|April-28-09:31:49|EfficientNet-B2|HorizontalFlip,VerticalFlip|384 * 384|16|0.7486| 0.65 
|April-28-12:52:12|EfficientNet-B3|HorizontalFlip,VerticalFlip|384 * 384|16|0.7571| 0.64
|April-28-16:45:22|EfficientNet-B4|HorizontalFlip,VerticalFlip|384 * 384|16|0.7468| 0.63
|April-28-16:45:22|EfficientNet-B5|HorizontalFlip,VerticalFlip|384 * 384|16|0.7475| 0.63
|April-29-10:39:11|EfficientNet-B2|crop_white, <br>HorizontalFlip,VerticalFlip,<br>RandomRotate90,<br>IAAAdditiveGaussianNoise,<br>GaussNoise,<br>RandomBrightnessContrast,<br>ShiftScaleRotate|512 * 512|16|0.8088|0.70
|April-29-10:39:11|EfficientNet-B3|same as above|512 * 512|16|0.7881|0.65|