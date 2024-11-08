from dataclasses import dataclass
from typing import List, Optional, Tuple
import typer
from rich import print
from rich.console import Console
from rich.table import Table
from rich.box import DOUBLE_EDGE
from pathlib import Path
from PIL import Image as PILImage
import os
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import json
from realesrgan import RealESRGANer
import sys
from contextlib import redirect_stdout
from io import StringIO

app = typer.Typer(help="GIF图片处理工具")
console = Console()

MODEL_CONFIGS = {
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "name": "RealESRGAN_x4plus.pth",
        "params": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 23,
            "num_grow_ch": 32,
            "scale": 4
        }
    },
    "realesrgan-x4plus-anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "name": "RealESRGAN_x4plus_anime_6B.pth",
        "params": {
            "num_in_ch": 3,
            "num_out_ch": 3,
            "num_feat": 64,
            "num_block": 6,
            "num_grow_ch": 32,
            "scale": 4
        }
    }
}

@dataclass
class GifInfo:
    """GIF图片信息的数据结构"""
    filename: str
    size: Tuple[int, int]
    mode: str
    frame_count: int
    duration: List[int]
    loop_count: Optional[int]
    file_size: float  # KB
    
    @property
    def fps(self) -> float:
        """计算平均帧率"""
        if not self.duration:
            return 0.0
        return 1000 / (sum(self.duration) / len(self.duration))
    
    @classmethod
    def from_file(cls, gif_path: Path) -> 'GifInfo':
        """从文件创建GifInfo实例"""
        with PILImage.open(gif_path) as img:
            # 获取帧信息
            frame_count = getattr(img, 'n_frames', 1)
            
            # 获取每一帧的持续时间
            duration = []
            for i in range(frame_count):
                img.seek(i)
                duration.append(img.info.get('duration', 0))

            # 循环次数
            loop_count = img.info.get('loop', None)
            if loop_count == 0:
                loop_count = None  # 表示无限循环

            return cls(
                filename=gif_path.name,
                size=img.size,
                mode=img.mode,
                frame_count=frame_count,
                duration=duration,
                loop_count=loop_count,
                file_size=os.path.getsize(gif_path) / 1024
            )

def version_callback(value: bool):
    if value:
        print("GIF处理工具 版本 1.0.0")
        raise typer.Exit()

@app.callback()
def callback(
    version: bool = typer.Option(False, "--version", callback=version_callback, help="显示版本信息"),
):
    pass

@app.command(name="view")
def view_command(
    gif_path: Path = typer.Argument(..., help="GIF文件路径", exists=True),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="显示详细信息"),
):
    """查看GIF信息"""
    try:
        with PILImage.open(gif_path) as img:
            # 计算文件大小
            file_size = gif_path.stat().st_size
            file_size_str = f"{file_size / 1024:.2f} KB"
            
            # 获取帧数
            n_frames = getattr(img, 'n_frames', 1)
            
            # 获取所有帧的延迟时间
            frame_delays = []
            for i in range(n_frames):
                img.seek(i)
                frame_delays.append(img.info.get('duration', 0))
            
            # 计算平均帧率
            avg_delay = sum(frame_delays) / len(frame_delays) if frame_delays else 0
            fps = f"{1000 / avg_delay:.2f} FPS" if avg_delay > 0 else "N/A"
            
            # 创建表格
            table = Table(title="GIF信息", box=DOUBLE_EDGE)
            table.add_column("属性", style="cyan")
            table.add_column("值", style="green")
            
            table.add_row("文件名", gif_path.name)
            table.add_row("图片大小", f"{img.width}x{img.height}")
            table.add_row("图片模式", "RGBA")  # wand总是使用RGBA
            table.add_row("总帧数", str(n_frames))
            table.add_row("帧持续时间", str(frame_delays))
            table.add_row("平均帧率", fps)
            table.add_row("循环次数", "无限")
            table.add_row("文件大小", file_size_str)
            
            console.print(table)
            
            if verbose:
                console.print("\n[yellow]详细帧信息:[/yellow]")
                frame_table = Table(box=DOUBLE_EDGE)
                frame_table.add_column("帧序号", style="cyan")
                frame_table.add_column("延迟(ms)", style="green")
                frame_table.add_column("尺寸", style="green")
                
                for i, frame in enumerate(img.sequence):
                    frame_table.add_row(
                        str(i + 1),
                        str(frame.delay),
                        f"{frame.width}x{frame.height}"
                    )
                console.print(frame_table)
                    
    except Exception as e:
        console.print(f"[red]错误: {str(e)}[/red]")
        raise typer.Exit(code=1)

def validate_model_name(model: str) -> None:
    """验证模型名称是否有效"""
    if model not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model}。支持的模型: {', '.join(MODEL_CONFIGS.keys())}")

def download_model(url: str, path: Path) -> None:
    """下载模型文件"""
    try:
        import requests
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        with Progress() as progress:
            task = progress.add_task("[yellow]下载模型...", total=total_size)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    progress.update(task, advance=size)
    except Exception as e:
        if path.exists():
            path.unlink()
        raise Exception(f"下载模型失败: {str(e)}")

def process_frame(frame: PILImage.Image, upsampler: RealESRGANer, scale: int) -> PILImage.Image:
    """处理单个帧"""
    # 转换为numpy数组
    np_frame = np.array(frame)
    
    # 使用RealESRGAN处理
    enhanced, _ = upsampler.enhance(np_frame, outscale=scale)
    
    # 清理内存
    del np_frame
    
    # 转换为PIL图像
    return PILImage.fromarray(enhanced)

# 在主循环中使用生成器
def process_frames(gif: PILImage.Image, upsampler: RealESRGANer, scale: int):
    """处理所有帧的生成器"""
    for i in range(gif.n_frames):
        gif.seek(i)
        frame = gif.convert('RGBA')
        yield process_frame(frame, upsampler, scale)

@app.command(name="upscale")
def upscale_command(
    gif_path: Path = typer.Argument(..., help="输入GIF文件的路径", exists=True),
    scale: int = typer.Option(2, "--scale", "-s", help="放大倍数 (2 或 4)", min=2, max=4),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="输出文件路径"),
    model: str = typer.Option(
        "realesrgan-x4plus", 
        "--model", 
        "-m",
        help="使用的模型 (realesrgan-x4plus 或 realesrgan-x4plus-anime)"
    ),
) -> None:
    """使用RealESRGAN放大GIF图片

    Args:
        gif_path: 输入GIF文件的路径
        scale: 放大倍数，支持2倍或4倍
        output: 输出文件路径，默认为原文件名加上_upscaled后缀
        model: 使用的模型，支持标准模型和动漫模型
    """
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import torch
        import numpy as np
        
        if output is None:
            output = gif_path.parent / f"{gif_path.stem}_upscaled{gif_path.suffix}"
            
        # 选择模型
        if model == "realesrgan-x4plus":
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            model_name = "RealESRGAN_x4plus.pth"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        else:  # realesrgan-x4plus-anime
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            model_name = "RealESRGAN_x4plus_anime_6B.pth"
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            
        # 创建模型目录
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / model_name
        
        # 下载模型（如果不存在）
        if not model_path.exists():
            console.print(f"[yellow]正在下载模型 {model_name}...[/yellow]")
            download_model(model_url, model_path)
        
        # 设置设备
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        console.print(f"[yellow]使用设备: {device}[/yellow]")
        
        # 使用重定向来抑制输出
        with redirect_stdout(open(os.devnull, 'w')):
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=device=='cuda',
                gpu_id=0 if device=='cuda' else None
            )
        
        # 禁用 upsampler 的打印功能
        upsampler.print_log = lambda *args, **kwargs: None
        
        # 修改 enhance 方法的行为
        original_enhance = upsampler.enhance
        def silent_enhance(*args, **kwargs):
            with redirect_stdout(open(os.devnull, 'w')):
                return original_enhance(*args, **kwargs)
        upsampler.enhance = silent_enhance

        # 读取原始GIF获取播放信息
        original_gif = PILImage.open(gif_path)
        n_frames = original_gif.n_frames
        
        # 获取所有帧的持续时间
        durations = []
        for i in range(n_frames):
            original_gif.seek(i)
            durations.append(original_gif.info['duration'])
        
        # 处理GIF
        enhanced_frames = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            refresh_per_second=10,  # 增加刷新率
            transient=False,  # 确保进度条不会消失
        ) as progress:
            task = progress.add_task("处理帧...", total=n_frames)
            
            for i in range(n_frames):
                original_gif.seek(i)
                frame = original_gif.convert('RGBA')
                
                if scale == 2:
                    enhanced, _ = upsampler.enhance(np.array(frame), outscale=2)
                else:  # scale == 4
                    enhanced, _ = upsampler.enhance(np.array(frame), outscale=4)
                
                pil_frame = PILImage.fromarray(enhanced)
                enhanced_frames.append(pil_frame)
                progress.update(task, advance=1)
        
        # 确保所有帧都被处理
        assert len(enhanced_frames) == n_frames, f"处理后的帧数 ({len(enhanced_frames)}) 与原始帧数 ({n_frames}) 不匹配"
        
        # 保存GIF，使用原始的帧持续时间
        enhanced_frames[0].save(
            str(output),
            save_all=True,
            append_images=enhanced_frames[1:],
            duration=durations,
            loop=original_gif.info.get('loop', 0),
            disposal=2,
            optimize=False
        )
        
        console.print(f"[green]已保存放大后的GIF到: {output}[/green]")
        
    except ImportError:
        console.print("[red]错误: 请先安装必要的依赖[/red]")
        console.print("pip install basicsr realesrgan torch numpy pillow")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[red]错误: {str(e)}[/red]")
        raise typer.Exit(code=1)

CONFIG_PATH = Path("config.json")

def load_config() -> dict:
    """加载配置文件"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}

def save_config(config: dict) -> None:
    """保存配置文件"""
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    app()