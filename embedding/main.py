import shutil

import gradio as gr

from config import *
from knowbase import KnowledgeBase


def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(KB_ROOT_PATH):
        return lst_default
    lst = os.listdir(KB_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


embedding_model_dict_list = list(embedding_model_dict.keys())
knowledge_base = KnowledgeBase()
knowledge_base.init_cfg()


def get_vector_store(vs_id, files, sentence_size, history, one_conent, one_content_segmentation):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    filelist = []
    if knowledge_base.embeddings:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
                filelist.append(os.path.join(KB_ROOT_PATH, vs_id, "content", filename))
            vs_path, loaded_files = knowledge_base.init_knowledge_vector_store(filelist, vs_path, sentence_size)
        else:
            vs_path, loaded_files = knowledge_base.one_knowledge_add(vs_path, files, one_conent,
                                                                     one_content_segmentation,
                                                                     sentence_size)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files if i])} 内容至知识库，并已加载知识库"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]], \
        gr.update(choices=knowledge_base.list_file_from_vector_store(vs_path) if vs_path else [])


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history, \
            gr.update(choices=[]), gr.update(visible=False)
    else:
        vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
        if "index.faiss" in os.listdir(vs_path):
            file_status = f"已加载知识库{vs_id}"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                vs_path, history + [[None, file_status]], \
                gr.update(choices=knowledge_base.list_file_from_vector_store(vs_path), value=[]), \
                gr.update(visible=True)
        else:
            file_status = f"已选择知识库{vs_id}，当前知识库中未上传文件，请上传文件"
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), \
                vs_path, history + [[None, file_status]], \
                gr.update(choices=[], value=[]), gr.update(visible=True, value=[])


def add_vs_name(vs_name, chatbot):
    if vs_name in get_vs_list():
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot, gr.update(visible=False)
    else:
        # 新建上传文件存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "content")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "content"))
        # 新建向量库存储路径
        if not os.path.exists(os.path.join(KB_ROOT_PATH, vs_name, "vector_store")):
            os.makedirs(os.path.join(KB_ROOT_PATH, vs_name, "vector_store"))
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储，请先完成文件上传"""
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=get_vs_list(), value=vs_name), gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot, gr.update(visible=True)


def refresh_vs_list():
    return gr.update(choices=get_vs_list()), gr.update(choices=get_vs_list())


def delete_file(vs_id, files_to_delete, chatbot):
    vs_path = os.path.join(KB_ROOT_PATH, vs_id, "vector_store")
    content_path = os.path.join(KB_ROOT_PATH, vs_id, "content")
    docs_path = [os.path.join(content_path, file) for file in files_to_delete]
    status = knowledge_base.delete_file_from_vector_store(vs_path=vs_path,
                                                          filepath=docs_path)
    if "fail" not in status:
        for doc_path in docs_path:
            if os.path.exists(doc_path):
                os.remove(doc_path)
    rested_files = knowledge_base.list_file_from_vector_store(vs_path)
    if "fail" in status:
        vs_status = "文件删除失败。"
    elif len(rested_files) > 0:
        vs_status = "文件删除成功。"
    else:
        vs_status = f"文件删除成功，知识库{vs_id}中已无文件，请上传文件"
    logger.info(",".join(files_to_delete) + vs_status)
    chatbot = chatbot + [[None, vs_status]]
    return gr.update(choices=knowledge_base.list_file_from_vector_store(vs_path), value=[]), chatbot


def delete_vs(vs_id, chatbot):
    try:
        shutil.rmtree(os.path.join(KB_ROOT_PATH, vs_id))
        status = f"成功删除知识库{vs_id}"
        logger.info(status)
        chatbot = chatbot + [[None, status]]
        return gr.update(choices=get_vs_list(), value=get_vs_list()[0]), gr.update(visible=True), gr.update(
            visible=True), \
            gr.update(visible=False), chatbot, gr.update(visible=False)
    except Exception as e:
        logger.error(e)
        status = f"删除知识库{vs_id}失败"
        chatbot = chatbot + [[None, status]]
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), \
            gr.update(visible=True), chatbot, gr.update(visible=True)


# 前端样式
block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""
default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)
webui_title = """
# 🎉Knowledge Base Home🎉
👍 [Using the text2vec-large-chinese Embedding Model](https://huggingface.co/GanymedeNil/text2vec-large-chinese)
"""
knowledge_base_mode_info = "您已进入知识库测试模式\n\n"

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    vs_path, file_status = gr.State(
        os.path.join(KB_ROOT_PATH, get_vs_list()[0], "vector_store") if len(get_vs_list()) > 1 else ""), gr.State(
        "")
    gr.Markdown(webui_title)
    with gr.Tab("知识库测试"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, knowledge_base_mode_info]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                # query = gr.Textbox(show_label=False,
                #                    placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["知识库测试"],  # "知识库问答",
                                label="请选择使用模式",
                                value="知识库测试",
                                visible=False)
                vs_setting = gr.Accordion("配置知识库", visible=True)
                with vs_setting:
                    vs_refresh = gr.Button("更新已有知识库选项")
                    select_vs = gr.Dropdown(get_vs_list(),
                                            label="请选择要加载的知识库",
                                            interactive=True,
                                            value=get_vs_list()[0] if len(get_vs_list()) > 0 else None)
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",
                                         lines=1,
                                         interactive=True,
                                         visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    vs_delete = gr.Button("删除本知识库", visible=False)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加单条内容或文件")
                        sentence_size = gr.Number(value=SENTENCE_SIZE, precision=0,
                                                  label="文本入库分句长度限制",
                                                  interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件",
                                            file_types=['.txt', '.md', '.docx', '.pdf'],
                                            file_count="multiple",
                                            show_label=False
                                            )
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件",
                                                   # file_types=['.txt', '.md', '.docx', '.pdf'],
                                                   file_count="directory",
                                                   show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                        with gr.Tab("添加单条内容"):
                            one_title = gr.Textbox(label="标题", placeholder="请输入要添加单条段落的标题", lines=1)
                            one_conent = gr.Textbox(label="内容", placeholder="请输入要添加单条段落的内容", lines=5)
                            one_content_segmentation = gr.Checkbox(value=True, label="禁止内容分句入库",
                                                                   interactive=True)
                            load_conent_button = gr.Button("添加内容并加载知识库")
                        with gr.Tab("删除文件"):
                            files_to_delete = gr.CheckboxGroup(choices=[],
                                                               label="请从知识库已有文件中选择要删除的文件",
                                                               interactive=True)
                            delete_file_button = gr.Button("从知识库中删除选中文件")
                    # 将上传的文件保存到content文件夹下,并更新下拉框
                    vs_refresh.click(fn=refresh_vs_list,
                                     inputs=[],
                                     outputs=select_vs)
                    vs_add.click(fn=add_vs_name,
                                 inputs=[vs_name, chatbot],
                                 outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    vs_delete.click(fn=delete_vs,
                                    inputs=[select_vs, chatbot],
                                    outputs=[select_vs, vs_name, vs_add, file2vs, chatbot, vs_delete])
                    select_vs.change(fn=change_vs_name_input,
                                     inputs=[select_vs, chatbot],
                                     outputs=[vs_name, vs_add, file2vs, vs_path, chatbot, files_to_delete, vs_delete])
                    load_file_button.click(get_vector_store,
                                           show_progress=True,
                                           inputs=[select_vs, files, sentence_size, chatbot, vs_add, vs_add],
                                           outputs=[vs_path, files, chatbot, files_to_delete], )
                    load_folder_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, folder_files, sentence_size, chatbot, vs_add,
                                                     vs_add],
                                             outputs=[vs_path, folder_files, chatbot, files_to_delete], )
                    load_conent_button.click(get_vector_store,
                                             show_progress=True,
                                             inputs=[select_vs, one_title, sentence_size, chatbot,
                                                     one_conent, one_content_segmentation],
                                             outputs=[vs_path, files, chatbot, files_to_delete], )
                    delete_file_button.click(delete_file,
                                             show_progress=True,
                                             inputs=[select_vs, files_to_delete, chatbot],
                                             outputs=[files_to_delete, chatbot])
    demo.load(
        fn=refresh_vs_list,
        inputs=None,
        outputs=[select_vs],
        # queue=True,
        show_progress=False,
    )

(demo
 # .queue(concurrency_count=3)  控制处理速率
 .launch(server_name='localhost',
         server_port=7861,
         show_api=False,
         share=False,
         inbrowser=False))
