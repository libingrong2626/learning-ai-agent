# 论文分类器 - 企业级项目版

这是按项目目录拆分后的版本。API Key 按要求保留为明文配置，分类规则保持不变。

## 运行方式

请在项目目录 `C:\Users\asus\Desktop\论文分类器` 下执行，程序会在当前目录生成 `印尼语论文分类结果`、`分类断点记录.json` 和 `logs`：

```powershell
python "C:\Users\asus\Desktop\论文分类器\run.py"
```

## 目录说明

- `run.py`：程序入口
- `paper_classifier/config`：配置和分类提示词
- `paper_classifier/common`：通用工具
- `paper_classifier/repositories`：原始论文读取
- `paper_classifier/infrastructure`：日志、断点、输出保存
- `paper_classifier/services`：API 调用和主流程编排

