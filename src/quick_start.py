# 1. 导入 MilvusClient
from pymilvus import MilvusClient, DataType
import random
import time
import os

# 2. 设置Hugging Face镜像源以加速模型下载（提供多个选项，取消注释选择一个）
# 清华大学镜像源
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 阿里云镜像源
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.aliyuncs.com'
# 中科大镜像源
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.ustc.edu.cn'

# 3. 导入 Sentence Transformer 模型用于文本转嵌入向量
try:
    from sentence_transformers import SentenceTransformer
    # 尝试加载模型，如果失败则使用随机向量
    try:
        print("正在加载嵌入模型... (如果首次运行，可能需要几分钟下载模型)")
        start_time = time.time()
        # 使用更小更快的模型
        model = SentenceTransformer('all-MiniLM-L6-v2')
        load_time = time.time() - start_time
        HAS_EMBEDDING_MODEL = True
        EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 的维度
        print(f"成功加载嵌入模型，耗时 {load_time:.2f} 秒")
    except Exception as e:
        print(f"加载嵌入模型失败: {e}，将使用随机向量")
        HAS_EMBEDDING_MODEL = False
        EMBEDDING_DIM = 384  # 保持维度一致
except ImportError:
    print("警告: 未安装 sentence-transformers 库，将使用随机向量")
    HAS_EMBEDDING_MODEL = False
    EMBEDDING_DIM = 384  # 使用较小的维度以提高性能

# 4. 创建一个 Milvus 客户端实例
#    这会在当前目录下创建一个名为 "milvus_demo.db" 的数据库文件。
#    如果该文件已存在，客户端会直接连接它。
client = MilvusClient("./milvus_demo.db")

# 5. 定义集合的 Schema
#    集合（Collection）类似于关系型数据库中的表。
#    这里我们定义一个包含 "id"（主键）和 "vector"（向量）字段的集合。
schema = client.create_schema(
    auto_id=False,  # 不自动生成 ID，我们将手动指定
    enable_dynamic_field=True,  # 允许动态添加其他字段
)

# 添加主键字段
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)

# 添加向量字段 (使用正确的维度)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

# 6. 创建索引参数
#    为向量字段创建索引可以极大地加速后续的相似性查询。
#    在本地模式下，只支持FLAT、IVF_FLAT和AUTOINDEX索引类型
index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="FLAT",  # 改为FLAT索引类型，本地模式支持
    metric_type="L2",  # 使用欧氏距离作为度量方式
    params={},  # FLAT索引不需要额外参数
)

# 7. 创建集合（如果已存在则先删除）
collection_name = "my_first_collection"
if client.has_collection(collection_name):
    print(f"集合 '{collection_name}' 已存在，正在删除...")
    client.drop_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    schema=schema,
    index_params=index_params,
)

print(f"集合 '{collection_name}' 创建成功。")

# 8. 文本转嵌入向量的函数
def text_to_embedding(text):
    """将文本转换为向量"""
    if HAS_EMBEDDING_MODEL:
        try:
            # 使用预训练模型生成嵌入向量
            embedding = model.encode(text)
            return embedding.tolist()  # 转换为列表格式
        except Exception as e:
            print(f"生成嵌入向量时出错: {e}，使用随机向量代替")
            return [random.random() for _ in range(EMBEDDING_DIM)]
    else:
        # 如果没有模型，则生成随机向量（保持兼容性）
        return [random.random() for _ in range(EMBEDDING_DIM)]

# 9. 插入文本数据
#    准备一些示例文本数据
texts_to_insert = [
    {"id": 0, "text": "人工智能是现代科技的重要发展方向"},
    {"id": 1, "text": "机器学习是实现人工智能的一种方法"},
    {"id": 2, "text": "深度学习是机器学习的一个分支"},
    {"id": 3, "text": "自然语言处理是人工智能的应用领域"},
    {"id": 4, "text": "计算机视觉让机器能够看懂图像"},
    {"id": 5, "text": "数据科学帮助我们理解大数据"},
    {"id": 6, "text": "Python是一种流行的编程语言"},
    {"id": 7, "text": "算法是程序设计的核心"},
    {"id": 8, "text": "云计算提供了强大的计算能力"},
    {"id": 9, "text": "物联网连接了各种智能设备"} ,
    {"id": 10, "text": "小明喜欢吃香蕉"},
    {"id": 11, "text": "小明喜欢画香蕉"},
    {"id": 12, "text": "小刚喜欢踢 ball"}
]

print("正在将文本转换为向量...")
# 将文本转换为向量并准备插入数据
vectors_to_insert = []
for i, item in enumerate(texts_to_insert):
    vector_data = {
        "id": item["id"],
        "vector": text_to_embedding(item["text"]),
        "text": item["text"]  # 保存原始文本以便检索时查看
    }
    vectors_to_insert.append(vector_data)
    if (i + 1) % 5 == 0:  # 每处理5个显示一次进度
        print(f"已处理 {i + 1}/{len(texts_to_insert)} 条文本")

# 执行插入操作
insert_result = client.insert(
    collection_name=collection_name,
    data=vectors_to_insert
)

print(f"成功插入 {insert_result['insert_count']} 条文本向量数据。")

# 10. 进行语义相似性查询
#    准备一个查询文本
query_text = "小明喜欢吃水果"
print(f"正在查询与 '{query_text}' 相似的文本...")
query_vector = text_to_embedding(query_text)

# 执行查询，寻找与 query_vector 最相似的 top 3 个向量
search_results = client.search(
    collection_name=collection_name,
    data=[query_vector],  # 可以一次性查询多个向量
    limit=10,  # 返回 top 10 的结果
    output_fields=["id", "text"],  # 指定要返回的字段
)

print(f"\n与查询文本 '{query_text}' 最相似的 10 个结果：")
# 按距离排序并显示结果
for hits in search_results:
    # Milvus默认已经按距离排序（升序），但为了确保，我们可以再次排序
    sorted_hits = sorted(hits, key=lambda x: x['distance'])
    for hit in sorted_hits:
        print(f"ID: {hit['id']}, 文本: {hit['entity']['text']}, 距离: {hit['distance']:.4f}")

# 11. (可选) 关闭客户端
#    在脚本结束时，客户端会自动关闭，所以这一步通常不是必须的。
#    client.close()