import gradio as gr
import os
import subprocess
import sys
from src.config import MODEL_DESCRIPTIONS, USAGE_TO_WEIGHTS_FILE, PRED_OUTPUT_DIR
from src.models import model_manager
from src.processors import process_single_image, process_batch_files, process_single_video
from src.image_processing import _resize_bg_keep_aspect, hex_to_rgb
from src.image_processing.utils import safe_progress
from src.utils.logger import logger, log_buffer

def build_semi_controls():
    with gr.Accordion("半透明优化 (针对婚纱、玻璃、烟雾等)", open=False):
        semi_enable = gr.Checkbox(label="扣除半透明 (V4.0 Unified)", value=False, info="开启后将智能识别并扣除半透明区域")
        
        with gr.Row(visible=False) as semi_group:
            semi_strength = gr.Slider(0, 1, 0.5, step=0.05, label="透明度调整 (Sensitivity)")
            
            # Helper text for the slider
            gr.Markdown(
                """
                <div style="font-size: 12px; color: #666; margin-top: -10px;">
                💡 调节指南：
                <ul>
                    <li><b>0.2 - 0.4</b>: 🛡️ <b>人像/实体</b> (保留更多主体，仅边缘半透)</li>
                    <li><b>0.5 - 0.6</b>: 🌫️ <b>薄纱/烟雾</b> (标准半透效果)</li>
                    <li><b>0.7 - 0.9</b>: ❄️ <b>玻璃/冰块/水</b> (强力去除背景色，高透亮)</li>
                </ul>
                </div>
                """
            )
            
        def toggle_semi(chk):
            return gr.update(visible=chk)
            
        semi_enable.change(toggle_semi, semi_enable, semi_group)
        
    return semi_enable, semi_strength

# Wrappers for UI callbacks to match Gradio inputs
def process_image_wrapper(
    image, 
    semi_enable, 
    semi_strength, 
    # semi_mode removed
    defringe, 
    defringe_strength,
    roi_enable, roi_editor, roi_meta, roi_crop_check, roi_pad,
    # resolution removed (redundant)
    _model_name_override=None
):
    # Get global resolution from model_manager (already updated by UI slider)
    resolution = 1024
    if hasattr(model_manager, 'current_loaded_resolution') and model_manager.current_loaded_resolution:
        resolution = model_manager.current_loaded_resolution[0]

    # ROI Logic
    roi_mask_fullres = None
    if roi_enable and roi_editor is not None and roi_meta is not None:
        try:
            from .components import _editor_layers_to_mask_fullres
            # We need to construct roi_layers from the editor output
            if isinstance(roi_editor, dict):
                roi_mask_fullres = _editor_layers_to_mask_fullres(roi_editor, roi_meta)
        except Exception as e:
            print(f"ROI Error: {e}")
            pass

    return process_single_image(
        image, 
        semi_enable, semi_strength, "auto", # Pass "auto" as dummy mode
        defringe, defringe_strength,
        _resolution=resolution,
        _roi_mask_fullres=roi_mask_fullres,
        _roi_crop_before=roi_crop_check,
        _roi_pad_px=roi_pad
    )

def process_batch_images_wrapper(
    files, 
    semi_enable, 
    semi_strength, 
    # semi_mode removed
    defringe, 
    defringe_strength,
):
    active_model = model_manager.current_loaded_model_name or "General"
    resolution = 1024 # Default for batch if not specified
    if hasattr(model_manager, 'current_loaded_resolution') and model_manager.current_loaded_resolution:
        resolution = model_manager.current_loaded_resolution[0] if isinstance(model_manager.current_loaded_resolution, tuple) else 1024

    return process_batch_files(
        files, 'image', 
        model_name=active_model, input_size=(int(resolution), int(resolution)),
        semi_enable=semi_enable, semi_strength=semi_strength, semi_mode="auto",
        defringe=defringe, defringe_strength=defringe_strength
    )

def process_batch_videos_wrapper(
    files, 
    resolution,
    bg_color,
    semi_enable, 
    semi_strength, 
    progress=gr.Progress()
):
    active_model = model_manager.current_loaded_model_name or "General"
    return process_batch_files(
        files, 'video', progress,
        model_name=active_model, input_size=(int(resolution), int(resolution)),
        bg_color=bg_color,
        semi_enable=semi_enable, semi_strength=semi_strength, semi_mode="auto"
    )

def create_interface():
    with gr.Blocks(
        title="小T的抠图工具箱",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        .tab-nav button {
            color: white !important;
            font-weight: bold !important;
        }
        .tab-nav button.selected {
            background: rgba(255,255,255,0.2) !important;
        }
        /* Smart ROI Editor Styling - Optimized for Gradio 5.x */
        .roi-editor-container {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background: #ffffff;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        /* Ensure the editor doesn't get squeezed */
        .roi-editor-container .gradio-image {
            min-height: 500px !important;
        }
        """
    ) as interface:
        
        # === 顶部标题区域（带退出按钮） ===
        with gr.Row():
            with gr.Column(scale=9):
                gr.Markdown(
                    """
                    # 🎯 小T的抠图工具箱
                    
                    **功能特点：**
                    - 🖼️ 支持单张图片和批量图片处理 (自动导出透明 PNG)
                    - 🎬 支持单个视频和批量视频处理 (支持自定义背景色/绿屏)
                    - 📦 批量处理结果自动打包下载
                    - ⚡ 高性能GPU加速推理
                    """
                )
            with gr.Column(scale=1, min_width=100):
                gr.Markdown("<br>")  # 添加一点垂直间距
                btn_exit_app = gr.Button("🚪 退出程序", variant="stop", size="sm")


        with gr.Accordion("⚙️ 模型与分辨率设置", open=True):
            # 生成带状态标记的模型选项
            def get_model_choices_with_status():
                status_map = model_manager.get_all_model_status()
                choices = []
                for key, desc in MODEL_DESCRIPTIONS.items():
                    status = status_map.get(key, "not_found")
                    if status == "ready":
                        icon = "✅"
                    elif status == "incomplete":
                        icon = "⚠️"
                    else:
                        icon = "⬇️"
                    choices.append(f"{key} - {desc} {icon}")
                return choices
            
            model_choices = get_model_choices_with_status()
            
            model_choice = gr.Dropdown(
                label="选择模型任务",
                choices=model_choices,
                value=model_choices[0],
                info="✅已下载 ⬇️需下载 ⚠️不完整"
            )

            resolution = gr.Slider(
                label="输入分辨率",
                minimum=256,
                maximum=2048,
                step=64,
                value=1024,
                info="设置模型推理输入分辨率"
            )
            resolution_info = gr.Markdown(
                value="⚙️ 当前输入分辨率：1024×1024\n💨 推理速度：中等（推荐）\n🎯 预估精度：高",
                label="分辨率性能提示"
            )
            status_box = gr.Textbox(label="状态", interactive=False)

            def on_model_change(selected_model):
                # Extract short name (remove status icon at end)
                short_name = selected_model.split(" - ")[0].strip()
                
                # 先检查模型状态
                local_status = model_manager.check_model_exists(short_name)
                if local_status == "ready":
                    status = f"📂 正在加载本地模型 {short_name} ..."
                elif local_status == "incomplete":
                    status = f"⚠️ 模型 {short_name} 不完整，尝试重新下载..."
                else:
                    status = f"⬇️ 正在下载模型 {short_name} (需联网)..."
                
                ok = model_manager.load_model(short_name, (1024, 1024))
                
                # 刷新下拉框选项以更新状态图标
                new_choices = get_model_choices_with_status()
                # 找到当前选中项的新值（带更新后的状态图标）
                new_value = next((c for c in new_choices if c.startswith(short_name + " - ")), new_choices[0])
                
                if ok:
                    status = f"✅ 模型加载成功：{short_name}"
                    
                    # Update resolution limits if needed
                    min_res, max_res = 256, 2048
                    default_value = 1024
                    if "lite-2K" in str(short_name):
                        min_res = 1024
                        return (status, 
                                gr.update(minimum=min_res, maximum=max_res, value=max(default_value, min_res), label="输入分辨率 (Lite 模型限制 ≥1024)"),
                                gr.update(choices=new_choices, value=new_value))
                    else:
                        return (status, 
                                gr.update(minimum=256, maximum=2048, value=1024, label="输入分辨率"),
                                gr.update(choices=new_choices, value=new_value))
                else:
                    status = f"❌ 模型加载失败：{short_name}\n请检查网络连接（需VPN访问HuggingFace）或手动导入模型"
                    return (status, 
                            gr.update(), 
                            gr.update(choices=new_choices, value=new_value))

            def on_resolution_change(res):
                res = int(res)
                # === 关键修复：更新全局分辨率状态 ===
                model_manager.current_loaded_resolution = (res, res)
                
                base_mem_gb = 2.5
                estimated_mem = base_mem_gb * (res / 1024) ** 2
                if res <= 512:
                    speed, quality = "🚀 非常快", "⚪ 精度较低"
                elif res <= 1024:
                    speed, quality = "⚡ 中等（推荐）", "🟢 精度高"
                elif res <= 1536:
                    speed, quality = "🐢 稍慢", "🔵 精度更高"
                else:
                    speed, quality = "🐌 较慢", "🟣 极高精度"
                
                return (
                    f"⚙️ 当前输入分辨率：{res}×{res}\n"
                    f"{speed} · {quality}\n"
                    f"🧠 预估显存占用：约 {estimated_mem:.1f} GB"
                )

            model_choice.change(fn=on_model_change, inputs=[model_choice], outputs=[status_box, resolution, model_choice])
            resolution.change(fn=on_resolution_change, inputs=[resolution], outputs=[resolution_info])

        with gr.Tabs():
            # Tab 1: Single Image
            with gr.Tab("🖼️ 单张图片"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="上传图片", type="numpy", sources=["upload", "clipboard", "webcam"], format="png", height=300)
                        
                        # ROI Controls
                        with gr.Accordion("🎯 区域抠图 (Smart ROI)", open=False):
                            with gr.Group(elem_classes="roi-editor-container"):
                                roi_enable = gr.Checkbox(label="启用区域抠图", value=False, info="开启后将根据您的涂抹笔迹保留对应物体")
                                
                                with gr.Row():
                                    btn_sync_roi = gr.Button("🔄 载入/重置编辑器图片", variant="secondary", size="sm")
                                    roi_pad = gr.Slider(0, 100, 16, label="背景提取外扩", visible=False) # Hide redundant slider
                                    roi_crop_check = gr.Checkbox(True, label="仅对该区域推理 (加速)", visible=False) # Keep hidden for now
                                
                                roi_editor = gr.ImageEditor(
                                    label="在图上涂抹感兴趣的物体", 
                                    type="numpy", 
                                    interactive=True,
                                    height=400, # Set explicit height for ROI editor
                                    brush=gr.Brush(colors=["#FF0000"], default_size=30),
                                    eraser=gr.Eraser(default_size=30),
                                    transforms=[], # Disable crop/rotate to focus on brush
                                    layers=False,  # Use single layer mode for better G5 compatibility if strokes are simple
                                    # G5 dynamic scaling fix
                                    sources=["upload", "clipboard"],
                                ) 
                                
                                def sync_to_editor(img):
                                    if img is None: return None
                                    return {"background": img, "layers": [], "composite": None}
                                
                                btn_sync_roi.click(sync_to_editor, input_image, roi_editor)

                        semi_enable, semi_strength = build_semi_controls()
                        
                        with gr.Accordion("边缘优化 (去除白边/Halo)", open=False):
                            defringe = gr.Checkbox(label="开启去白边", value=False)
                            defringe_strength = gr.Slider(0, 1, 0.7, label="去白边强度")
                        
                        btn_run = gr.Button("🚀 开始生成", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="处理结果", type="numpy", format="png", height=300)
                        mask_preview = gr.Image(label="Mask 预览", type="numpy", format="png", height=300)

                # State helper
                # ROI meta state needs to capture original size.
                roi_meta_state = gr.State()
                
                def update_roi_meta(img):
                    if img is None: return None
                    h, w = img.shape[:2]
                    # Editor defaults to resizing, but we need original size.
                    # Actually Gradio ImageEditor handles resizing internally.
                    # We need to reimplement _make_editor_thumbnail logic if we want to control it carefully,
                    # or just trust Gradio's editor value full res if type='numpy'.
                    # If type='numpy', background is the array.
                    # But the editor canvas might be resized.
                    # Let's simplify and assume the editor returns full res layers matching background if we passed full res background.
                    return {"ori_w": w, "ori_h": h, "thumb_w": w, "thumb_h": h} # Simplified

                input_image.change(update_roi_meta, input_image, roi_meta_state)

                btn_run.click(
                    fn=process_image_wrapper,
                    inputs=[
                        input_image, 
                        semi_enable, semi_strength, 
                        defringe, defringe_strength,
                        roi_enable, roi_editor, roi_meta_state, roi_crop_check, roi_pad,
                    ],
                    outputs=[output_image, mask_preview]
                )

            # Tab 2: Batch Images
            with gr.Tab("📚 批量图片"):
                files_input = gr.File(label="上传多张图片", file_count="multiple", file_types=["image"])
                
                b_semi_enable, b_semi_strength = build_semi_controls()
                
                with gr.Accordion("边缘优化 (去除白边/Halo)", open=False):
                    b_defringe = gr.Checkbox(label="开启去白边", value=False)
                    b_defringe_strength = gr.Slider(0, 1, 0.7, label="去白边强度")
                
                btn_batch_run = gr.Button("🚀 批量处理", variant="primary")
                batch_output = gr.File(label="下载结果 (ZIP)")
                batch_msg = gr.Textbox(label="状态")

                btn_batch_run.click(
                    fn=process_batch_images_wrapper,
                    inputs=[
                        files_input,
                        b_semi_enable, b_semi_strength, 
                        b_defringe, b_defringe_strength
                    ],
                    outputs=[batch_output, batch_msg]
                )

            # Tab 3: Video
            with gr.Tab("🎬 视频处理"):
                video_input = gr.Video(label="上传视频")
                v_bg_color = gr.ColorPicker(label="背景颜色", value="#00FF00")
                
                v_semi_enable, v_semi_strength = build_semi_controls()
                
                btn_video_run = gr.Button("🚀 处理视频", variant="primary")
                video_output = gr.Video(label="结果视频")
                video_msg = gr.Textbox(label="状态")

                btn_video_run.click(
                    fn=lambda v, c, se, ss, res: process_single_video(v, model_manager.current_loaded_model_name, (int(res),int(res)), c, None, se, ss, "auto"), # Simplified lambda
                    inputs=[
                        video_input, v_bg_color,
                        v_semi_enable, v_semi_strength, 
                        resolution
                    ],
                    outputs=[video_output, video_msg]
                )

            # Tab 4: Batch Video
            with gr.Tab("🎞️ 批量视频"):
                v_files_input = gr.File(label="上传多个视频", file_count="multiple", file_types=["video"])
                vb_color = gr.ColorPicker(label="背景颜色", value="#00FF00")
                
                vb_semi_enable, vb_semi_strength = build_semi_controls()
                
                btn_vb_run = gr.Button("🚀 批量处理视频", variant="primary")
                vb_output = gr.File(label="下载结果 (ZIP)")
                vb_msg = gr.Textbox(label="状态")

                btn_vb_run.click(
                    fn=process_batch_videos_wrapper,
                    inputs=[
                        v_files_input, resolution, vb_color,
                        vb_semi_enable, vb_semi_strength
                    ],
                    outputs=[vb_output, vb_msg]
                )

            # Tab 5: 日志查看
            with gr.Tab("📜 运行日志"):
                gr.Markdown("查看程序运行日志，方便排查错误。日志保留最近 500 条记录。")
                with gr.Row():
                    btn_refresh_log = gr.Button("🔄 刷新日志", variant="secondary")
                    btn_clear_log = gr.Button("🧹 清空日志", variant="secondary")
                
                log_display = gr.Textbox(
                    label="运行日志",
                    value=log_buffer.get_logs,
                    lines=20,
                    max_lines=30,
                    interactive=False,
                    show_copy_button=True
                )
                
                def refresh_log():
                    return log_buffer.get_logs()
                
                def clear_log():
                    log_buffer.clear()
                    logger.info("日志已清空")
                    return log_buffer.get_logs()
                
                btn_refresh_log.click(fn=refresh_log, outputs=[log_display])
                btn_clear_log.click(fn=clear_log, outputs=[log_display])

        # === 底部实用功能区 ===
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                btn_open_output = gr.Button("📂 打开输出文件夹", variant="secondary")
            with gr.Column(scale=1):
                btn_release_cache = gr.Button("🧹 释放显存/内存", variant="secondary")
            with gr.Column(scale=2):
                with gr.Row():
                    # 动态获取已下载模型列表
                    def get_downloaded_model_choices():
                        downloaded = model_manager.get_downloaded_models()
                        if not downloaded:
                            return ["（暂无已下载模型）"]
                        return downloaded
                    
                    delete_model_dropdown = gr.Dropdown(
                        label="选择要删除的模型",
                        choices=get_downloaded_model_choices(),
                        interactive=True,
                        scale=2
                    )
                    btn_delete_model = gr.Button("🗑️ 删除模型", variant="stop", scale=1)
        
        utility_status = gr.Textbox(label="操作状态", interactive=False, visible=True)
        
        # === 功能按钮回调 ===
        def open_output_folder():
            try:
                if sys.platform == 'win32':
                    os.startfile(PRED_OUTPUT_DIR)
                elif sys.platform == 'darwin':  # macOS
                    subprocess.run(['open', PRED_OUTPUT_DIR])
                else:  # Linux
                    subprocess.run(['xdg-open', PRED_OUTPUT_DIR])
                return f"✅ 已打开输出文件夹：{PRED_OUTPUT_DIR}"
            except Exception as e:
                return f"❌ 打开文件夹失败：{e}"
        
        def release_cache():
            if model_manager.unload_model():
                # 刷新模型选择下拉框状态
                new_choices = get_model_choices_with_status()
                return "✅ 模型已卸载，显存/内存已释放。下次处理图片时会自动重新加载模型。", gr.update(choices=new_choices, value=new_choices[0])
            else:
                return "ℹ️ 当前没有加载的模型", gr.update()
        
        def delete_selected_model(model_name):
            if not model_name or model_name == "（暂无已下载模型）":
                return "⚠️ 请先选择要删除的模型", gr.update()
            
            if model_manager.delete_model(model_name):
                # 刷新下拉框选项
                new_choices = get_downloaded_model_choices()
                new_model_choices = get_model_choices_with_status()
                return (f"✅ 模型 {model_name} 已删除", 
                        gr.update(choices=new_choices, value=new_choices[0] if new_choices else None),
                        gr.update(choices=new_model_choices, value=new_model_choices[0]))
            else:
                return f"❌ 删除模型 {model_name} 失败", gr.update(), gr.update()
        
        btn_open_output.click(fn=open_output_folder, outputs=[utility_status])
        btn_release_cache.click(fn=release_cache, outputs=[utility_status, model_choice])
        btn_delete_model.click(
            fn=delete_selected_model, 
            inputs=[delete_model_dropdown], 
            outputs=[utility_status, delete_model_dropdown, model_choice]
        )
        
        # === 退出按钮回调 ===
        def exit_application():
            """退出应用程序"""
            import time
            import threading
            
            def delayed_exit():
                time.sleep(0.5)  # 延迟0.5秒，让UI有时间显示消息
                
                # 调用app.py中的退出函数
                try:
                    # 通过__main__模块访问app.py中的函数
                    import __main__
                    if hasattr(__main__, 'shutdown_application'):
                        __main__.shutdown_application()
                    else:
                        # 如果找不到函数，直接退出
                        import os
                        os._exit(0)
                except Exception as e:
                    logger.error(f"退出时出错: {e}")
                    import os
                    os._exit(0)
            
            # 在后台线程执行退出，避免阻塞UI响应
            threading.Thread(target=delayed_exit, daemon=True).start()
            
            return "✅ 程序即将退出，请稍候..."
        
        btn_exit_app.click(fn=exit_application, outputs=[utility_status])


    return interface

