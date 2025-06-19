# MQM-APE Analyzer / MQM-APE 分析器

A Python tool for analyzing MQM-APE (Multi-dimensional Quality Metric - Automatic Post-Editing) translation quality evaluation results. This analyzer processes JSON data from MQM-APE evaluations and provides detailed statistical analysis, visualizations, and normalized scoring.

一个用于分析 MQM-APE 翻译质量评估结果的 Python 工具。该分析器处理来自 MQM-APE 评估的 JSON 数据，提供详细的统计分析、可视化和标准化评分。

## Overview / 概述

MQM-APE (Lu et al., 2024) is a framework that enhances translation quality evaluation by:
- Identifying translation errors using Large Language Models (LLMs)
- Applying automatic post-editing to correct identified errors
- Verifying which errors actually improve translation quality

MQM-APE 是一个通过以下方式增强翻译质量评估的框架：
- 使用大型语言模型（LLMs）识别翻译错误
- 应用自动后编辑来纠正识别出的错误
- 验证哪些错误实际上能提高翻译质量

This analyzer processes the results from MQM-APE evaluations and provides comprehensive analysis tools for researchers, practitioners, and **translation educators** working with translation quality assessment. It is particularly well-suited for **translation teaching and student translation quality assessment**, enabling instructors to:

该分析器处理 MQM-APE 评估的结果，为从事翻译质量评估的研究人员、从业者和**翻译教育工作者**提供全面的分析工具。它特别适合**翻译教学和学生译文质量评估**，使教师能够：

- **Evaluate student translations systematically** / **系统地评估学生译文**
- **Provide detailed error feedback with visual reports** / **提供带有可视化报告的详细错误反馈**
- **Compare student performance across multiple assignments** / **比较学生在多个作业中的表现**

[![image](picture or gif url)](https://github.com/nalkalin/MQM-APE_Analyzer/blob/main/mqm_charts.png?raw=true)

## Features / 功能特性

- **Normalized Scoring** / **标准化评分**: Calculate normalized penalty scores across multiple students/tasks / 计算多个学生/任务的标准化惩罚分数
- **Student Performance Tracking** / **学生表现跟踪**: Individual and comparative assessment across multiple assignments / 多个作业的个人和比较评估
- **Error Analysis** / **错误分析**: Extract and categorize translation errors by severity and type / 按严重程度和类型提取和分类翻译错误
- **Visualizations** / **可视化**: Generate charts including: / 生成图表，包括：
  - Nested pie charts for error distribution by severity and category / 按严重程度和类别显示错误分布的嵌套饼图
  - Error type distribution bar charts / 错误类型分布条形图
  - Score distribution histograms / 分数分布直方图
  - Quality score box plots for class performance analysis / 用于班级表现分析的质量分数箱线图
- **Excel Export** / **Excel 导出**: Export comprehensive analysis results to multi-sheet Excel files for grading / 将综合分析结果导出到多工作表 Excel 文件用于评分
- **Statistical Analysis** / **统计分析**: Detailed statistics with score normalization and scaling / 包含分数标准化和缩放的详细统计
- **Customizable Parameters** / **可定制参数**: Adjustable segments-per-task for different evaluation scenarios / 针对不同评估场景的可调整每任务片段数

## Installation / 安装

### Requirements / 依赖要求

```bash
pip install pandas numpy matplotlib json os collections warnings openpyxl
```

### Dependencies / 依赖项

- `pandas` - Data manipulation and analysis / 数据操作和分析
- `numpy` - Numerical computations / 数值计算
- `matplotlib` - Visualization and plotting / 可视化和绘图
- `json` - JSON data handling / JSON 数据处理
- `openpyxl` - Excel file operations / Excel 文件操作

## Usage / 使用方法

### Basic Usage / 基本使用

```python
from mqm_ape_analyzer import MQMAPEAnalyzer

# Initialize the analyzer / 初始化分析器
analyzer = MQMAPEAnalyzer("results.json", segments_per_task=4)

# Print comprehensive summary / 打印综合摘要
analyzer.print_summary()

# Generate visualizations / 生成可视化图表
analyzer.create_visualizations("mqm_charts.png")

# Export to Excel / 导出到 Excel
analyzer.export_to_excel("mqm_analysis.xlsx")
```

### Translation Teaching Example / 翻译教学示例

```python
# Example for evaluating student translation assignments
# 评估学生翻译作业的示例

# Initialize analyzer for a class of 8 students, each with 4 translation segments
# 为8名学生的班级初始化分析器，每人有4个翻译片段
analyzer = MQMAPEAnalyzer("student_translations.json", segments_per_task=4)

# Generate individual student reports / 生成个别学生报告
analyzer.print_summary()

# Create visual feedback for students / 为学生创建可视化反馈
analyzer.create_visualizations("class_performance_charts.png")

# Export detailed grading report for instructor / 为教师导出详细评分报告
analyzer.export_to_excel("student_assessment_report.xlsx")

# Get specific statistics for grading / 获取用于评分的具体统计信息
stats = analyzer.get_statistics()
print(f"Class Average: {stats['Current Scaled Quality Score (0-100)']:.1f}/100")
```

### Advanced Configuration / 高级配置

```python
# Custom segments per task (adjust based on your evaluation setup)
# 自定义每任务片段数（根据您的评估设置调整）
analyzer = MQMAPEAnalyzer(
    json_file_path="results.json",
    segments_per_task=6  # Modify based on your TQA requirements
                        # 根据您的TQA要求修改
)

# Get detailed statistics / 获取详细统计信息
stats = analyzer.get_statistics()
print(f"Current Quality Score: {stats['Current Scaled Quality Score (0-100)']:.1f}/100")

# Create custom analysis tables / 创建自定义分析表
detailed_table = analyzer.create_detailed_table()
summary_table = analyzer.create_summary_table()
```

## Input Format / 输入格式

The analyzer expects a JSON file with the following structure:

分析器期望一个具有以下结构的 JSON 文件：

```json
[
  {
    "source_seg": "Source text in original language",
    "target_seg": "Translation text",
    "source_lang": "zh",
    "target_lang": "en", 
    "MQM_APE_score": -5.0,
    "error_dict": {
      "critical": [],
      "major": [
        {
          "category": "accuracy/mistranslation",
          "span": "error text",
          "post_edit": "corrected text"
        }
      ],
      "minor": [
        {
          "category": "fluency/grammar", 
          "span": "error text",
          "post_edit": "corrected text"
        }
      ]
    }
  }
]
```

## Output Files / 输出文件

### Visualization (`mqm_charts.png`) / 可视化图表

- **Nested Pie Chart** / **嵌套饼图**: Error distribution by severity (outer ring) and category (inner ring) / 按严重程度（外环）和类别（内环）的错误分布
- **Bar Chart** / **条形图**: Top 10 error types by frequency / 按频率排列的前10种错误类型
- **Histogram** / **直方图**: Distribution of original MQM scores / 原始 MQM 分数的分布
- **Box Plot** / **箱线图**: Quality score distribution across students/tasks / 学生/任务间的质量分数分布

### Excel Export (`mqm_analysis.xlsx`) / Excel 导出

Contains multiple worksheets: / 包含多个工作表：

- **Detailed Analysis** / **详细分析**: Complete error information with normalized scores / 包含标准化分数的完整错误信息
- **Error Summary** / **错误摘要**: Statistical summary by error severity and type / 按错误严重程度和类型的统计摘要
- **Quality Metrics** / **质量指标**: Overall performance statistics / 整体性能统计
- **Students Scores** / **学生分数**: Individual student performance with normalized scores / 个人学生表现及标准化分数
- **Normalization Info** / **标准化信息**: Score calculation metadata / 分数计算元数据

## Key Concepts / 核心概念

### Score Normalization / 分数标准化

The analyzer implements a sophisticated scoring methodology: / 分析器实现了一套巧妙的评分方法：

1. **Student Grouping** / **学生分组**: Segments are grouped by `segments_per_task` parameter / 按 `segments_per_task` 参数对片段进行分组
2. **Penalty Calculation** / **惩罚计算**: Normalized penalty scores calculated per student / 计算每个学生的标准化惩罚分数
3. **Quality Scaling** / **质量缩放**: Scores scaled to 0-100 range based on relative performance / 基于相对表现将分数缩放到 0-100 范围
4. **Cross-Student Comparison** / **跨学生比较**: Enables fair comparison across different tasks/students / 实现不同任务/学生间的公平比较

### Error Categories / 错误类别

Supports standard MQM error categories: / 支持标准 MQM 错误类别：

- **Accuracy** / **准确性**: mistranslation, omission, addition / 错译、遗漏、添加
- **Fluency** / **流畅性**: grammar, punctuation, spelling, register / 语法、标点、拼写、语域
- **Style** / **文体**: awkward phrasing / 措辞别扭
- **Terminology** / **术语**: inappropriate or inconsistent usage / 不当或不一致的使用

### Severity Levels / 严重程度级别

- **Critical** / **严重**: Errors that inhibit comprehension / 阻碍理解的错误
- **Major** / **重要**: Errors that disrupt flow but maintain comprehensibility / 破坏流畅性但保持可理解性的错误
- **Minor** / **轻微**: Technical errors that don't hinder understanding / 不妨碍理解的技术性错误

## Configuration Options / 配置选项

### Segments Per Task / 每任务片段数

Adjust the `segments_per_task` parameter based on your TQA setup:

根据您的TQA设置调整 `segments_per_task` 参数：

```python
# For different TQA configurations
# 针对不同的TQA配置
analyzer = MQMAPEAnalyzer("results.json", segments_per_task=8)  # 8 segments per task / 每任务8个片段
analyzer = MQMAPEAnalyzer("results.json", segments_per_task=2)  # 2 segments per task / 每任务2个片段
```

### Visualization Customization / 可视化定制

The analyzer uses customizable color schemes: / 分析器使用可定制的颜色方案：

- **Severity Colors** / **严重程度颜色**: Red (Critical), Orange (Major), Blue (Minor) / 红色（严重）、橙色（重要）、蓝色（轻微）
- **Category Colors** / **类别颜色**: Pink (Accuracy), Cyan (Fluency), Magenta (Style), etc. / 粉色（准确性）、青色（流畅性）、洋红色（文体）等

## Example Output / 示例输出

### Summary Statistics / 摘要统计

```
MQM-APE Translation Quality Analysis / MQM-APE 翻译质量分析
====================================
Total Translation Segments: 16 / 翻译片段总数：16
Number of Students: 4 / 学生数量：4
Current Scaled Quality Score (0-100): 78.5 / 当前缩放质量分数（0-100）：78.5
Total Error Count: 12 / 错误总数：12
Average Errors per Segment: 0.75 / 每片段平均错误数：0.75

Student Performance Summary / 学生表现摘要:
   Student 1: 85.2/100 (Excellent) / 学生1：85.2/100（优秀）
   Student 2: 78.5/100 (Good) / 学生2：78.5/100（良好）
   Student 3: 72.1/100 (Fair) / 学生3：72.1/100（一般）
   Student 4: 69.8/100 (Needs Improvement) / 学生4：69.8/100（需要改进）
```

### Quality Metrics / 质量指标

- Normalized penalty scores for fair comparison / 用于公平比较的标准化惩罚分数
- Scaled quality scores (0-100) for intuitive interpretation / 便于直观理解的缩放质量分数（0-100）
- Cross-student performance analysis / 跨学生表现分析
- Error distribution preservation / 错误分布保持
- MT system performance benchmarking / MT系统性能基准测试

## Use Cases / 使用场景

- **Translation Teaching & Education** / **翻译教学与教育**: 
  - Student TQA / 学生译文质量评估
  - Automated grading and feedback generation / 自动评分和反馈生成
  - Progress tracking across multiple assignments / 跟踪多个作业的进步情况
  - Comparative analysis of student performance / 学生表现的比较分析
- **MT System Evaluation** / **MT系统评估**: Comprehensive evaluation of MT system performance / MT 系统性能的综合评估
- **Comparative TQA** / **比较TQA**: Compare different translation systems or approaches / 比较不同的翻译系统或方法
- **Error Pattern Analysis** / **错误模式分析**: Identify common error types and severity distributions / 识别常见错误类型和严重程度分布
- **Professional TQA** / **专业TQA**: Assess translation quality in commercial settings / 在商业环境中评估翻译质量
- **Research Applications** / **研究应用**: Academic research in MT evaluation / MT评估的学术研究

## Technical Notes / 技术说明

### Error Handling / 错误处理

The analyzer includes error handling for: / 分析器包含针对以下情况的错误处理：

- Malformed JSON input files / 格式错误的 JSON 输入文件
- Missing required fields in MQM-APE output / MQM-APE输出中缺少必需字段
- Invalid score ranges / 无效的分数范围
- Empty datasets / 空数据集

## Contributing / 贡献

This tool is designed for research and practical applications in TQA. For issues, feature requests, or contributions, please refer to the project repository.

该工具专为TQA的研究和实际应用而设计。如有问题、功能请求或贡献，请参考项目仓库。

## Bibliography / 参考文献

This project is based on the MQM-APE framework:

本项目基于 MQM-APE 框架：

```bibtex
@article{lu2024mqm,
  title={MQM-APE: Toward High-Quality Error Annotation Predictors with Automatic Post-Editing in LLM Translation Evaluators},
  author={Lu, Qingyu and Ding, Liang and Zhang, Kanjian and Zhang, Jinxia and Tao, Dacheng},
  journal={arXiv preprint arXiv:2409.14335},
  year={2024}
}
```
