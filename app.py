import gradio as gr
from version_1.car_error import tab1_ui
from version_2.model2 import tab2_ui
from version_3_openAI.tab3 import tab3_ui

with gr.Blocks() as app:
    with gr.Tab("석현이네 정비센터"):
        tab1_ui()
    with gr.Tab("명우네 정비센터"):
        tab2_ui()
    with gr.Tab("OpenAI 정비센터"):
        tab3_ui()


app.launch()
