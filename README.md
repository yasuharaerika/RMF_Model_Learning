# RMF_Model_Learning
相对论平均场模型学习及实现

分为三个步骤
PART1 walecka模型
Walecka_Model.py

PART2 密度依赖模型
DDME.py
包含DD-MEX和DD-ME2两组参数，及对称核物质（delta=0）和纯中子物质（delta=1）。

PART3 中子星物质模型
DDME_beta.py
增加了beta-平衡相关函数，通过XX函数来选择是否出现beta平衡
并修改了函数XX来

DDME_NS.py
默认为beta-平衡条件下的中子星物质。
主要生成状态方程（P-epsilon）关系，并求解对应质量半径关系。
PS：需调用function_set.py，为之前其他工作写的函数总包，主要调用其中TOV方程求解类，默认有壳层。
