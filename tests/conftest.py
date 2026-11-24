"""pytest 的公共配置。

这个文件名是 pytest 硬编码的约定：运行前会自动加载同目录下的 conftest.py，
不需要任何 import。它的作用范围是"当前目录及所有子目录"。

没有它的话，在别的目录直接敲 pytest 时，`import app` 会找不到包而报错。
"""

import sys
from pathlib import Path

# __file__ = .../chat-service/tests/conftest.py
# parents[0] = tests/   parents[1] = chat-service/（项目根）
ROOT = Path(__file__).resolve().parents[1]

# 把项目根插到 sys.path 最前面，Python 找模块时才会从那里开始找 app 包
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))