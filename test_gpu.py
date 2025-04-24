import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import time
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

##########################################  streamlit run test.py 前端界面
# 页面样式配置
st.set_page_config(
    page_title="LemonAI 智能助手",
    page_icon="🍋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #FDFFBC 0%, #FFEAC8 100%);
    }
    .stChatMessage {
        padding: 15px 20px;
        border-radius: 18px !important;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .user .stChatMessage {
        background: #F8FFD0;
        border: 1px solid #E0E9A3;
    }
    .assistant .stChatMessage {
        background: #FFF9E6;
        border: 1px solid #F5E3A9;
    }
    .lemon-header {
        font-family: 'Microsoft YaHei';
        color: #5E8B2C;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .fas {
        font-family: 'Font Awesome 5 Free' !important;
    }
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css');
</style>
""", unsafe_allow_html=True)

# 侧边栏配置
with st.sidebar:
    # LOGO
    st.image("https://cdn-icons-png.flaticon.com/512/3199/3199018.png", width=80)

    # 对话设置
    st.divider()
    st.markdown("**🍋 对话设置  🐖**")
    temperature = st.slider("创意思考能力", 0.0, 1.0, 0.3,
                            help="创造度越高，越适合开放性的问题")

    # 知识库设置
    st.divider()
    st.markdown("**🌳 知识库设置**")
    search_depth = st.slider("检索深度", 1, 18, 3,
                             help="调整参考文档的数量")
    confidence_threshold = st.slider("置信阈值", 0.5, 1.0, 0.7,
                                     help="设置知识匹配的置信度要求")

    # 知识库更新功能
    st.divider()
    st.markdown("**📚 知识库更新**")
    uploaded_file = st.file_uploader("上传新文档（支持 TXT/PDF）", type=["txt", "pdf"])
    if st.button("更新知识库", type="primary"):
        if uploaded_file is not None:
            with st.spinner("正在更新知识库..."):
                try:
                    st.session_state.qa_system.update_knowledge_base(uploaded_file)
                    st.success("知识库更新成功！")
                except Exception as e:
                    st.error(f"知识库更新失败: {str(e)}")
        else:
            st.warning("请上传一个 TXT 或 PDF 文件。")

    st.divider()
    st.markdown("**🔄 知识库重置**")
    if st.button("重置到初始知识库", type="primary", help="将知识库恢复到原始比赛文档状态"):
        with st.spinner("正在恢复初始知识库..."):
            try:
                import shutil
                import os

                # 路径配置
                FAISS_INDEX_PATH = "./faiss_index_data"
                INITIAL_BACKUP_PATH = "./initial_faiss_backup"

                # 删除当前索引
                if os.path.exists(FAISS_INDEX_PATH):
                    shutil.rmtree(FAISS_INDEX_PATH)

                # 从备份恢复初始索引
                shutil.copytree(INITIAL_BACKUP_PATH, FAISS_INDEX_PATH)

                # 重新初始化系统
                st.session_state.qa_system.initialize_components()
                st.toast("🍋 知识库已恢复到初始状态!", icon="✅")
            except Exception as e:
                st.error(f"恢复失败: {str(e)}")

    if st.button("清空对话历史", type="primary"):
        st.session_state.messages = [
            {"role": "assistant", "content": "hi~我是LemonAI，支持比赛信息的知识库问答，有什么可以帮您呢？"}
        ]
        st.rerun()

# 顶部标题栏
header_col1, header_col2, header_col3 = st.columns([0.15, 0.7, 0.15])
with header_col1:
    st.image("https://cdn-icons-png.flaticon.com/512/3199/3199018.png", width=80)
with header_col2:
    st.markdown('<h1 class="lemon-header">🍋🐖 LemonPigAI 比赛答疑智能助手</h1>', unsafe_allow_html=True)
with header_col3:
    st.markdown("""
    <div style="text-align: right; margin-top: 15px;">
        <span style="font-size: 0.9em; color: #888; font-style: italic;">
            <i class="fas fa-lightbulb" style="color: #FFD700;"></i>
            🍋 如果回答不满意或不确定建议您再提问一次，我会根据上文再次做出优化哦! 🐖 
        </span>
    </div>
    """, unsafe_allow_html=True)

# 消息历史处理
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "hi~我是LemonAI，支持比赛信息的知识库问答，有什么可以帮您呢？"}
    ]

# 配置参数
QWEN_MODEL_NAME = "./LLM_large"
EMBEDDING_MODEL_PATH = "./vector_model"
FAISS_INDEX_PATH = "./faiss_index_data"


# 提示模板
def load_enhanced_prompt():
    return PromptTemplate(
        template="""### 参考内容：
{context}
### 用户问题：
{question}
### 回答要求：
1. 基于参考内容总结，禁止复制原文
2. 分点列出核心信息（1. 2. 3.）
3. 每个要点<30字，总回答<100字
4. 标注引用来源[数字]
5. 无相关信息时说明
### 专业回答：""",
        input_variables=["context", "question"]
    )



############################################################    知识库检索匹配部分
# 智能问答系统
class EnhancedQASystem:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.embeddings = None
        self.create_initial_backup()

    def initialize_components(self):
        # 加载嵌入模型
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_PATH,
            model_kwargs={'device': 'cuda'}
        )

        # 加载向量库
        self.vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

        # 初始化语言模型
        llm = self.load_qwen_model()

        # 构建 QA 链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": search_depth}),
            chain_type_kwargs={"prompt": load_enhanced_prompt()},
            return_source_documents=True
        )

    def load_qwen_model(self):
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=500,
            temperature=0.4,
            top_p=0.85,
            repetition_penalty=1.2,
            device_map="auto"
        )
        return HuggingFacePipeline(pipeline=llm_pipeline)

    # 计算文档匹配相似度，这里我们采用余弦相似度
    def get_similarity_scores(self, query, documents):
        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = self.embeddings.embed_documents([doc.page_content for doc in documents])
        similarity_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
        return similarity_scores

    # 更新知识库的方法
    def update_knowledge_base(self, uploaded_file):
        # 创建临时目录保存上传文件
        import tempfile
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)

        # 保存上传文件到临时目录
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 判断文件类型并选择加载器
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.lower().endswith(('.txt', '.text')):
            loader = TextLoader(file_path)
        else:
            raise ValueError("Unsupported file type")

        # 加载和处理文档
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # 更新向量库
        self.vector_store.add_documents(docs)
        self.vector_store.save_local(FAISS_INDEX_PATH)

        # 重新构建 QA 链
        llm = self.load_qwen_model()
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": search_depth}),
            chain_type_kwargs={"prompt": load_enhanced_prompt()},
            return_source_documents=True
        )

        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Error cleaning temp files: {str(e)}")

    def create_initial_backup(self):
        """创建初始知识库备份"""
        FAISS_INDEX_PATH = "./faiss_index_data"
        INITIAL_BACKUP_PATH = "./initial_faiss_backup"

        if not os.path.exists(INITIAL_BACKUP_PATH):
            try:
                shutil.copytree(FAISS_INDEX_PATH, INITIAL_BACKUP_PATH)
                print("Initial knowledge base backup created.")
            except Exception as e:
                print(f"Backup creation failed: {str(e)}")


# 初始化知识库系统
if "qa_system" not in st.session_state:
    with st.spinner("知识库系统初始化中..."):
        st.session_state.qa_system = EnhancedQASystem()
        st.session_state.qa_system.initialize_components()




################################################################    二次经过模型，进行推理
@st.cache_resource(show_spinner="lemonAI正在出炉...")
def load_model():
    model_path = "LLM_large"
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map='cuda',
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None, None


model, tokenizer = load_model()


# 自定义生成函数（整合知识库回答和答案简化）
def generate_response(prompt, history):
    # 因为要返回两个值
    lemon_keywords_1 = ["你好", "hi", "hello"]
    if any(keyword in prompt for keyword in lemon_keywords_1):
        return "你好~ 🍋我能帮你更清楚的了解一些比赛信息，有什么想问的嘛,请直接问问题哦，不用再和我打招呼了~", ""
    lemon_keywords_2 = ["拜拜", "bye"]
    if any(keyword in prompt for keyword in lemon_keywords_2):
        return "我随时在树上再等你提问~ 再会~ 🍋", ""
    lemon_keywords_3 = ["情感", "感情"]
    if any(keyword in prompt for keyword in lemon_keywords_3):
        return "🍋 也喜欢你，比赛加油~", ""

    # 通过知识库系统生成专业回答
    with st.spinner("🍋 正在检索知识库..."):
        try:
            kb_result = st.session_state.qa_system.qa_chain({"query": prompt})
            lemon_answer = kb_result["result"]
            source_documents = kb_result["source_documents"]
            similarity_scores = st.session_state.qa_system.get_similarity_scores(prompt, source_documents)
            max_similarity = max(similarity_scores)

            if (max_similarity < 0.5):
                st.toast("检测到您的问题与知识库匹配度不足", icon="🍋")
                return "您确定这个问题和这次的比赛知识库有关嘛，我不太清楚答案欸，您再调整一下询问方法再试试呗 ~ ", ""

            with st.expander("查看问题与知识库的匹配相似度", expanded=False):
                for i, doc in enumerate(source_documents):
                    st.markdown(f"文档 {i + 1} 相似度: {similarity_scores[i]:.3f}")
        except Exception as e:
            lemon_answer = "知识库检索失败，请重试"

    # 组合简化 prompt
    simplify_prompt = f"用户问题：{prompt}\n专业回答：{lemon_answer}\n要求：根据问题找到答案并简要概括，一定要精准，如果你你觉得答案不匹配问题，你可以自己再根据问题加一点点开放性的泛化答案"
    # 对话模型生成简化回答
    with st.status("🍋 正在树上生成答案中...", expanded=False) as status:
        time.sleep(0.5)
        messages = history + [{"role": "user", "content": simplify_prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt")

            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=500,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(
                outputs[0][len(inputs.input_ids[0]):],
                skip_special_tokens=True
            )
            status.update(label="🍋 思考完成！", state="complete")
            return response, lemon_answer
        except Exception as e:
            return f"报意思 🍋 还不够成熟，思考失败…… ", ""



##################################################  主面板
# 主界面逻辑
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("输入您的问题吧..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成知识库回答和简化回答
    simplified_answer, kb_answer = generate_response(prompt, st.session_state.messages[:-1])

    # 添加对话历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": simplified_answer,
        "kb_answer": kb_answer
    })

    with st.chat_message("assistant"):
        st.markdown(simplified_answer)

    # 显示知识库原始回答
    if kb_answer:
        with st.expander("查看知识库详细回答", expanded=False):
            st.markdown(f"**知识库专业回答：**\n{kb_answer}")

