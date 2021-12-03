Pytorch 文档:

Tensor: 类似numpy的ndarray, 能够使用GPU加速运算, x.size() 查看shape

Difference between tensor and matrix:	

​	tensor: 张量, 标量视为零阶张量，矢量视为一阶张量，那么矩阵就是二阶张量。

常见operation: torch.add(x, y)  y.add_(x)    (x and y are tensor)





For part (c) interpret all of the regression coefficients. What does it mean that they have the signs they do (if anything). 
In interpreting an interaction term (like the location by seat belt interaction Agresti asks about), 
figure out what the size of the effects are for all four classes (for this interaction, rural without seat belt, rural with seat belt, urban without seat belt, urban with seat belt).
Estimated odds ratio equals exp(−0.7602 − 0.1244) = 0.41 in rural locations and
exp(−0.7602) = 0.47 in urban locations. The interaction effect -0.1244 is the difference
between the two log odds ratios.
