# PANDA

|Date|Model| Task |Augmentation |Image Size| Iuput Size |Validation | LB |
|----|-----|------|------|------------|------------|-----------|----|
|April-27-16:28:23|EfficientNet-B0|Regression|HorizontalFlip,VerticalFlip|384 * 384|16|0.7287| 0.64
|April-27-23:24:32|EfficientNet-B1|Regression|HorizontalFlip,VerticalFlip|384 * 384|16|0.7588| 0.62
|April-28-09:31:49|EfficientNet-B2|Regression|HorizontalFlip,VerticalFlip|384 * 384|16|0.7486| 0.65 
|April-28-12:52:12|EfficientNet-B3|Regression|HorizontalFlip,VerticalFlip|384 * 384|16|0.7571| 0.64
|April-28-16:45:22|EfficientNet-B4|Regression|HorizontalFlip,VerticalFlip|384 * 384|16|0.7468| 0.63
|April-28-16:45:22|EfficientNet-B5|Regression|HorizontalFlip,VerticalFlip|384 * 384|16|0.7475| 0.63
|April-29-10:39:11|EfficientNet-B2|Regression|crop_white, <br>HorizontalFlip,VerticalFlip,<br>RandomRotate90,<br>IAAAdditiveGaussianNoise,<br>GaussNoise,<br>RandomBrightnessContrast,<br>ShiftScaleRotate|512 * 512|16|0.8088|0.70
|April-29-10:39:11|EfficientNet-B3|Regression|same as above|512 * 512|16|0.7881|0.65|
|May-01-20:45:45|Se-Resnext50_32x4d|Classification|same as above|tile size: 128, num tiles: 12|24|0.8209|0.76|
|May-03-14:39:48|Se-Resnext50_32x4d|Classification|RandomRotate90<br>Flip<br>Transpose<br>IAAAdditiveGaussianNoise<br>GaussNoise<br>MotionBlur<br>MedianBlur<br>Blur<br>ShiftScaleRotate<br>HueSaturationValue|tile size: 128, num tiles: 12|24|0.8070,0.7898,0.8294,0.8203,0.8051|0.77|
|May-03-15:16:00|Se-Resnext101_32x4d|Classification|same as above|tile size: 128, num tiles: 12|24|0.8030,0.7784,0.8197,0.8072,0.8021|0.76|
|May-03-15:16:00|Se-Resnext50_32x4d|Regression|same as above|tile size: 128, num tiles: 12|24|0.8097,0.7955,0.8214,0.8179,0.7873|0.79|
|May-04-15:00:00|Se-Resnext50_32x4d|Regression|enhance_image<br>crop_white<br>same as above|tile size: 128, num tiles: 12|24|0.8087,0.7937,0.8145,0.8245,0.7902|0.80|
|May-04-15:00:00|Se-Resnext50_32x4d|Regression|enhance_image<br>crop_white<br>same as above|tile size: 128, num tiles: 16|16|0.7992,0.7870,0.8156,0.8140,0.7909|0.79|