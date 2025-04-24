from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm


class PDFStructureParser:
    def __init__(self, skip_first=True):
        self.learned_title_style = None
        self.current_section = None
        self.content_buffer = []
        self.sections = []
        self.skip_first = skip_first  # 跳过标记
        self.first_title_processed = False  # 首个标题处理标记

    def _get_text_style(self, element):
        """获取精确文本样式特征"""
        font_sizes = []
        fonts = []
        bolds = []
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                for char in text_line:
                    if isinstance(char, LTChar):
                        font_sizes.append(char.size)
                        fonts.append(char.fontname)
                        bolds.append('Bold' in char.fontname)
        return {
            'avg_size': sum(font_sizes) / len(font_sizes) if font_sizes else 0,
            'main_font': max(set(fonts), key=fonts.count) if fonts else None,
            'is_bold': any(bolds)
        }
    def _clean_text(self, text):
        """文本清洗"""
        text = re.sub(r'[\n\r]+', ' ', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\u3000', ' ', text)
        text = re.sub(r'[]', '', text)
        return text.strip()
    def _is_first_level_title(self, text, style):
        """判断是否为一级标题（动态学习+格式匹配）"""
        # 首次匹配时学习样式
        if not self.learned_title_style:
            if self._match_title_format(text):
                self.learned_title_style = style
                return True
            return False

        # 后续匹配：格式+样式双重验证
        return (
                self._match_title_format(text)
                and self._is_style_similar(style, self.learned_title_style)
        )

    def _match_title_format(self, text):
        """匹配一级标题格式"""
        patterns = [
            r'^第[一二三四五六七八九十]+条\s+',  # 条款式标题：第一条
            r'^[一二三四五六七八九十]、\s*',  # 中文顿号标题：一、
            r'^Chapter\s+\d+',  # 英文标题：Chapter 1
            r'^[\dA-Z]+\.\s+[^\d]+$'  # 混合标题：1.1 Introduction
        ]
        return any(re.match(p, text) for p in patterns)

    def _is_style_similar(self, style1, style2):
        """样式相似性判断（允许±1.5pt误差）"""
        return (
                abs(style1['avg_size'] - style2['avg_size']) <= 1.5
                and style1['main_font'] == style2['main_font']
                and style1['is_bold'] == style2['is_bold']
        )

    def parse(self, filepath):
        """解析逻辑"""
        for page_layout in extract_pages(filepath):
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    raw_text = element.get_text().strip()
                    if not raw_text:
                        continue

                    text = self._clean_text(raw_text)
                    style = self._get_text_style(element)

                    if self._is_first_level_title(text, style):
                        # 处理首个标题跳过逻辑
                        if self.skip_first and not self.first_title_processed:
                            self.first_title_processed = True
                            self._flush_content_buffer(skip=True)  # 跳过首个内容
                            continue

                        self._flush_content_buffer()
                        self.current_section = text
                    else:
                        self.content_buffer.append(text)

        self._flush_content_buffer()
        return self.sections
    def _flush_content_buffer(self, skip=False):
        """提交缓冲区（新增跳过逻辑）"""
        if self.content_buffer and not skip:
            self.sections.append({
                'content': '\n'.join(self.content_buffer),
                'section': self.current_section  # 保存当前章节信息
            })
        self.content_buffer = []

exclude_indexes=[3]
class TextProcessor:
    def __init__(self ):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            separators=["\n\n", "。", "！", "？", "；"]
        )
        self.exclude_indexes = exclude_indexes

    def process(self, sections, filename, file_index):
        """处理分块（新增条件判断）"""
        all_chunks = []
        for section in sections:
            # 获取清洗后内容
            cleaned = re.sub(r'\s+', ' ', section['content'])

            # 添加前缀条件判断
            prefix = ""
            if not (file_index in self.exclude_indexes and section == sections[0]):
                prefix += f"：{section['section']}：\n" if section['section'] else ""

            # 添加文件名前缀
            prefix = f"{os.path.splitext(filename)[0]}\n"+prefix

            # 分块处理
            chunks = self.text_splitter.split_text(cleaned)
            processed_chunks = [prefix + chunk for chunk in chunks]
            print(processed_chunks)
            all_chunks.extend(processed_chunks)

        return all_chunks


def generate_vectors(persist_dir, all_texts, embeddings):
    """生成并保存向量库"""
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)
    # 始终创建新向量库（无元数据）
    vectorstore = FAISS.from_texts(
        texts=all_texts,
        embedding=embeddings
    )

    vectorstore.save_local(persist_dir)


def process_pdf_folder(pdf_folder, persist_directory):
    """处理整个PDF文件夹"""
    embeddings = HuggingFaceEmbeddings(
        model_name="vector_model",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


    processor = TextProcessor()

    all_texts = []

    pdf_files = sorted([f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')])
    for idx, filename in enumerate(tqdm(pdf_files, desc="处理PDF文件")):
        pdf_path = os.path.join(pdf_folder, filename)
        skip_first = idx + 1 not in exclude_indexes  # 索引从1开始
        parser = PDFStructureParser(skip_first=skip_first)
        sections = parser.parse(pdf_path)
        chunks = processor.process(sections, filename, file_index=idx + 1)
        all_texts.extend(chunks)

    # 生成向量库
    generate_vectors(persist_directory, all_texts, embeddings)



if __name__ == "__main__":
    # 配置路径
    pdf_folder = './pdf_folder'
    persist_directory = 'faiss_index_data'

    # 处理PDF并生成向量库
    process_pdf_folder(pdf_folder, persist_directory)

    print(f"\n处理完成！向量库已保存到 {persist_directory}")