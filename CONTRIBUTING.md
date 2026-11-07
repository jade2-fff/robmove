# 协作开发

仓库管理员先在 GitHub 仓库的 **Settings → Collaborators** 中邀请协作者。协作者接受邀请后，使用自己的 GitHub 账号克隆仓库、推送分支和创建 Pull Request（PR）。账号登录使用 GitHub CLI、个人访问令牌或 SSH 密钥。

## 提交代码

首次参与时，克隆仓库并进入项目目录：

```bash
git clone https://github.com/jade2-fff/robmove.git
cd robmove
```

每次开始修改前，在项目根目录执行：

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/my-change
```

完成修改后，运行离线检查：

```bash
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release
```

如果修改了 `tools/chest_yolo_debug_config.py` 或对应配置，还需安装 PyYAML，并运行：

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
```

检查修改内容，暂存需要提交的文件，再提交和推送：

```bash
git status
git diff
git add path/to/changed-file
git diff --cached
git commit -m "feat: describe the change"
git push -u origin feature/my-change
```

将示例文件路径和分支名替换为实际值，然后在 GitHub 页面创建 PR，目标分支选择 `main`。PR 中写明修改目的、主要变化和验证结果，等待检查通过及维护者审核后合并。

## 上传范围

- 提交源码、项目配置、测试、文档及项目需要的模型和 CAD 资源。
- 编译产物、`work/` 调试记录、虚拟环境和本机工具配置已通过 `.gitignore` 排除。
- 登录凭据、访问令牌和私钥不得提交；本机环境变量可放在已忽略的 `.env` 文件中。
- `pick_place_pipeline/` 和原始资源受 `migration_manifest.json` 完整性校验保护。相关变更需说明原因，并同步核实校验记录和对应测试。

构建依赖和工程结构见 [README](README.md)，模块边界见 [架构说明](docs/architecture.md)。现有 GitHub Actions 会在 push 和 PR 时执行 Release 与 ASan 构建及离线测试。
