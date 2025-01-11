本项目代码是2024深圳大学计算机前沿技术课程的项目复现代码。
所有代码均由（作者：曾培根 2450101009） 实现。

***

#### 文件说明
其中：
- auto.py: 包含运行其他代码所使用的各种自动化模块。
- arch.py: 包含GLA Transformer架构和其他所需架构组件。
- env_ar_test.py: 包含进行associated recall训练测试的模块。
- env_text.py: 包含进行自回归训练的模块。
- gated_att.cu: 为GLA Cuda Kernel的Cpp代码。
- script_ar.py: 为进行associated recall测试的脚本。
- script_text.py: 为进行文本预训练的脚本。
- script_text_ask_only.py: 为使用已训练的一个模型进行推理的脚本。
- 0x7fc046eb7cd0.pth: 训练得到的模型文件。
- simple_book_92_train.txt: 训练所使用的小说文本。

#### 运行

本项目代码使用依赖版本为：
torch 2.1
ninja 1.11

运行 script_ar.py 进行associated recall测试

运行 script_text.py 进行文本预训练

运行 script_text_ask_only.py 可使用一个训练好的模型进行推理
