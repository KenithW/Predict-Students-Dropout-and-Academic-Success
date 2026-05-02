# 基于学生退学与学业成功数据的机器学习预测研究（中文报告草稿）

## 摘要

本研究使用 UCI Machine Learning Repository 中的 Predict Students' Dropout and Academic Success 数据集，探索机器学习模型能否预测学生最终学业状态，并进一步分析早期学业表现是否能显著提升预测能力。数据集包含 4424 名高等教育学生的入学信息、人口统计特征、社会经济因素以及第一、第二学期学业表现。目标变量 `Target` 为三分类结果：`Dropout`、`Enrolled` 和 `Graduate`。

本文的核心设计是比较不同预测时间点的信息价值：第一组模型只使用入学时可获得的背景和申请信息；第二组模型加入第一学期表现；第三组模型加入第一和第二学期表现。该设计能够区分“入学时早期预警”和“学期表现已知后的风险识别”，避免将未来学业表现错误地当作入学前可用信息。模型比较包括 DummyClassifier、Logistic Regression、Decision Tree、KNN、SVM、Random Forest 和 MLP，并使用 Accuracy、Macro F1、Weighted F1 和 per-class recall 评价模型表现。由于类别不平衡，本文将 Macro F1 作为主要指标，并重点关注 `Dropout` 类别的 recall。

## 1. 引言与研究问题

高等教育中的学生退学问题会影响学生个人发展、学校资源配置和教学支持策略。若能够较早识别退学风险较高的学生，学校可以更及时地提供学术辅导、经济支持或心理支持。因此，学生退学与学业成功预测是一个具有现实意义的机器学习应用场景。

本文研究问题为：

**机器学习模型能否预测学生最终是退学、仍在读还是毕业？与仅使用入学信息相比，加入第一和第二学期学业表现能否显著提升预测效果？**

子问题包括：

1. 哪类机器学习模型最适合预测 `Dropout`、`Enrolled` 和 `Graduate`？
2. 从入学信息到第一、第二学期表现，预测性能如何变化？
3. 类别不平衡处理是否能改善少数类或重点类别的识别？
4. 哪些变量对退学风险和学业成功预测最重要？

## 2. 数据与预处理

数据通过 `ucimlrepo.fetch_ucirepo(id=697)` 读取。数据集共有 4424 条样本和 36 个特征，无缺失值。目标变量 `Target` 包含三个类别：`Graduate` 约占一半，`Dropout` 约占三分之一，`Enrolled` 占比最小，因此存在类别不平衡。

本文将特征分为三个集合：

- **Enrollment-only**：入学时已知的 24 个变量，包括申请方式、课程、入学成绩、年龄、社会经济指标、学费状态、奖学金状态等。
- **+ 1st semester**：在入学变量基础上加入 6 个第一学期课程表现变量。
- **+ 1st + 2nd semester**：加入全部 36 个特征，包括第一和第二学期表现。

类别编码变量使用 one-hot encoding，数值变量使用标准化处理。所有预处理步骤均放入 pipeline，并且所有调参只在训练集上完成，以避免测试集信息泄漏。

## 3. 探索性数据分析

EDA 首先检查目标类别分布，说明 `Graduate` 是最大类，`Enrolled` 是最小类，因此不能只依赖 accuracy 评价模型。随后分析年龄、入学成绩和先前学历成绩在不同目标类别中的分布，并比较不同课程的目标组成。

早期学业表现分析重点关注第一和第二学期已通过课程数量、课程成绩等变量。通常情况下，退学学生在已通过课程数和学期成绩上会明显低于毕业学生，这为后续模型性能提升提供了直观解释。

## 4. 方法

本文比较以下分类模型：

| 模型 | 作用 |
|---|---|
| DummyClassifier | 多数类基线 |
| Logistic Regression | 可解释的线性分类模型 |
| Decision Tree | 简单树模型 |
| KNN | 距离型分类器 |
| SVM (RBF) | 非线性间隔分类器 |
| Random Forest | 集成树模型 |
| MLP | 简单神经网络分类器 |

模型选择阶段使用 5-fold stratified cross-validation，并通过 `GridSearchCV` 进行小范围调参。主要指标为 Macro F1，因为它平等看待三个类别，能避免模型只偏向 `Graduate` 类。Accuracy 和 Weighted F1 作为辅助指标；per-class recall 用于分析模型是否能识别 `Dropout` 学生。

## 5. 结果与解释

结果部分应报告三个特征集合下的最佳模型表现，并重点比较 Macro F1 和 `Dropout` recall。预期上，Enrollment-only 模型性能最低，但它代表最早可部署的预警场景；加入第一学期表现后，模型应明显改善；加入第二学期表现后通常会进一步提高，因为这些变量与最终学业状态更接近。

类别不平衡实验比较 Logistic Regression、class weight、random oversampling 和 Random Forest 等策略。讨论重点不是某个方法是否让 accuracy 最高，而是它是否改善 `Dropout` 或 `Enrolled` 的 recall，以及是否牺牲了其他类别表现。

可解释性部分使用 Logistic Regression 的 `Dropout` 类系数和 Random Forest feature importance。第一、第二学期通过课程数和成绩可能成为重要预测变量；学费是否按时缴纳、奖学金状态、年龄、课程类型和入学成绩也可能具有解释价值。所有解释均为预测关联，不应直接解释为因果关系。

## 6. 讨论

本研究的主要价值在于明确区分不同预测时间点。若目标是在学生入学时就进行风险预警，则不能使用第一或第二学期成绩；若目标是在学期结束后进行更准确的干预，则学期表现变量是合理且有价值的输入。这个设计可以很好地展示 data leakage 和 real-world deployment timing 的区别。

研究限制包括：

- 数据来自一个高等教育机构，模型泛化到其他国家或学校前需要重新验证。
- `Enrolled` 类别样本较少，可能导致该类别 recall 较低。
- 特征重要性反映预测关联，不代表因果关系。
- 使用人口统计和社会经济变量时需要考虑公平性和伦理问题，避免模型加剧已有不平等。

## 7. 结论

本文基于 UCI 学生退学与学业成功数据集完成了一个完整的机器学习三分类研究。项目比较了多个课堂模型、不同预测时间点和类别不平衡处理策略，并通过混淆矩阵、per-class recall 和特征重要性分析解释模型表现。整体而言，该数据集非常适合展示 COMP4030 coursework 要求的数据科学流程，包括问题定义、EDA、预处理、模型比较、调参、评估、解释性分析和局限性讨论。

## 参考文献

[1] V. Realinho, M. Vieira Martins, J. Machado, and L. Baptista, "Predict Students' Dropout and Academic Success," UCI Machine Learning Repository, 2021. doi: 10.24432/C5MC89.

[2] F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.

[3] H. He and E. A. Garcia, "Learning from Imbalanced Data," IEEE Transactions on Knowledge and Data Engineering, vol. 21, no. 9, pp. 1263-1284, 2009.
