# Python Milvus Lite 示例项目

这是一个使用 Milvus Lite 和小型 LLM 模型构建的向量搜索示例项目。该项目演示了如何将文本转换为向量、存储到 Milvus 向量数据库中，并使用小型 LLM 模型处理搜索结果。

## 功能特性

- 使用 Sentence Transformers 将文本转换为向量
- 利用 Milvus Lite 存储和检索向量数据
- 集成小型 LLM 模型处理搜索结果
- 支持多线程并行下载模型
- 使用阿里云镜像源加速模型下载

## 环境要求

- Python 3.7+
- Milvus Lite
- sentence-transformers
- transformers (用于 LLM 模型)

## 安装依赖

```bash
pip install pymilvus sentence-transformers transformers
```

## 配置说明

项目中已配置以下优化选项：

1. **模型下载镜像源**：
   - 默认使用阿里云镜像源加速 Hugging Face 模型下载
   - 可切换至清华大学或中科大镜像源

2. **并行下载**：
   - 启用 8 线程并行下载模型文件

3. **模型信任设置**：
   - 设置 `trust_remote_code=True` 以允许执行远程代码

## 使用方法

运行示例代码：

```bash
python src/quick_start.py
```

该脚本将执行以下操作：
1. 加载嵌入模型和小型 LLM 模型
2. 创建 Milvus 集合并插入示例数据
3. 执行语义相似性查询
4. 使用 LLM 模型处理并生成自然语言回答

## 代码结构

- `src/quick_start.py` - 主程序文件，包含完整的向量搜索和 LLM 处理流程

## 注意事项

- 首次运行时需要下载模型文件，可能需要几分钟时间
- GPT-2 是英文模型，对中文处理效果有限
- 可替换为更适合中文的小型模型以获得更好效果