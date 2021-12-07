**前置知识: Python迭代器，生成器**

- 迭代器: https://www.bilibili.com/video/BV1BT4y1P7nn?from=search&seid=11200523217797380099&spm_id_from=333.337.0.0
  - 迭代器类型的定义:
    - 当类中定义了```__iter___```和```__next__```两个方法
    - ```__iter```方法需要返回对象本身，即: self
    - ```__next__```方法，返回下一个数据，如果没有数据，抛出StopInteration的异常
    - 例子： for循环依赖于迭代器
    - ![image-20211206154532230](/Users/qile/Library/Application Support/typora-user-images/image-20211206154532230.png)
  - 迭代器对象:
    - 根据迭代器类实例化的对象
- 生成器:
  - 生成器函数: 
    - ```def func(): yield1 yield2```
    - Obj1 = func(), 创建生成器对象(内部是根据生成器generator创建的对象,不用自己写方法), 生成器类的内部也声明了: ```__iter__```, ```__next__```方法，一种特殊的迭代器，编写方式，表现形式不同
- 可迭代对象:
  - 如果一个类中有```__iter__```方法且返回一个迭代器对象：则我们称这个类创建的对象为可迭代对象，![image-20211206154427877](/Users/qile/Library/Application Support/typora-user-images/image-20211206154427877.png)
  - ![image-20211206160547837](/Users/qile/Library/Application Support/typora-user-images/image-20211206160547837.png)
  - ![image-20211206160813459](/Users/qile/Library/Application Support/typora-user-images/image-20211206160813459.png)
  - ![image-20211206160309995](/Users/qile/Library/Application Support/typora-user-images/image-20211206160309995.png)
- 可迭代对象可以通过增加类方法实现更多的功能，比如list，它是个可迭代对象，但是它的功能远远超出迭代器所有的，比如append，clear，copy等等。迭代器实质上只是一个强大的类的配件
- 实验：![image-20211206163148208](/Users/qile/Library/Application Support/typora-user-images/image-20211206163148208.png)
- ![image-20211206163200476](/Users/qile/Library/Application Support/typora-user-images/image-20211206163200476.png)
- 可以看到生成器对象就是一种迭代器对象，而list这种为可迭代对象（Iterable），要调用```__iter()__```返回迭代器对象(Iterator)。



**Tutorial:**

Pytorch 文档: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html



1. **处理数据**: 
   1. torch.utils.data.Dataset:
      1.  stores the samples and their corresponding labels
   2. torch.utils.data.DataLoader:
      1. wraps an iterable around the `Dataset`
      2. supports automatic batching, sampling, shuffling and multiprocess data loading. 

2. **Creating Models:**
   1. Define the layers of the Network in the ```__init__``` function and specify how data will pass through the network in the ```forward()``` function.
   2. device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")；model.to(device)
      1. 这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
3. **Optimizing the Model Parameters**:
   1. ![image-20211206165918761](/Users/qile/Library/Application Support/typora-user-images/image-20211206165918761.png)

4. **Saving Models:**
   1. ![image-20211206170140581](/Users/qile/Library/Application Support/typora-user-images/image-20211206170140581.png)
5. **Loading Models:**
   1. ![image-20211206170155320](/Users/qile/Library/Application Support/typora-user-images/image-20211206170155320.png)



**Tensor:**

Tensor: 类似numpy的ndarray, 能够使用GPU加速运算, x.size() 查看shape

Difference between tensor and matrix:	

​	tensor: 张量, 标量视为零阶张量，矢量视为一阶张量，那么矩阵就是二阶张量。

- matmul和mul的区别: matmul是tensor a*b, mul是a和b逐位相乘
- 取出single-element tensor的值: agg = tensor.sum(), agg_item = agg.item(), 使用```.item()```
- In-place operation: tensor.add_(5)
- print(f"t: {t}"): 加f =》字符串格式化，可以再字符串里面使用花括号括起来的变量和表达式
- **注:** 如果np.array转tensor, 内存地址相同，修改一个另一个也会变
- 常见operation: torch.add(x, y)  y.add_(x)    (x and y are tensor)



**Transforms**:

**transform:** to modify the features

**target_transform:** to modify the labels - that accept callables containing the transformation logic.



**Build the Neural Network**: 

```
print("Model structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```





**AUTOMATIC DIFFERENTIATION WITH TORCH.AUTOGRAD**:

- we need to be able to compute the gradients of loss function with respect to those variables. In order to do that, we set the `requires_grad` property of those tensors. example: requires_grad=True

- ```
  with torch.no_grad(): 不需要反向传播训练时
  ```



**Optimizer:**

```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Reset the gradients of model parameters: optimizer.zero_grad()

- Call `optimizer.zero_grad()` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
- Backpropagate the prediction loss with a call to `loss.backward()`. PyTorch deposits the gradients of the loss w.r.t. each parameter.
- Once we have our gradients, we call `optimizer.step()` to adjust the parameters by the gradients collected in the backward pass.
- ![image-20211206195347202](/Users/qile/Library/Application Support/typora-user-images/image-20211206195347202.png)

![image-20211206195447712](/Users/qile/Library/Application Support/typora-user-images/image-20211206195447712.png)
