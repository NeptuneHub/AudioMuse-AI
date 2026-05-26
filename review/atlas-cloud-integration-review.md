# Atlas Cloud Provider Review

## 已完成改动

- 新增 `ATLAS` 作为独立 AI provider，而不是复用 `OPENAI` 名称混配，便于 setup、chat、clustering 和后续维护。
- 在后端调度层接入 Atlas Cloud：`tasks/ai/api.py` 已支持 `ATLAS` 配置校验、文本生成和 tool-calling，底层复用现有 OpenAI-compatible transport。
- 在配置层新增 Atlas 环境变量：`ATLAS_SERVER_URL`、`ATLAS_MODEL_NAME`、`ATLAS_API_KEY`。
- 在聊天页和聚类页新增 Atlas Cloud provider 选项与对应 URL / Model 输入项。
- 在 `README.md` 新增 Atlas Cloud 介绍、官方链接、coding plan 文案，以及 Atlas logo 展示。
- 已把 logo 复制到仓库资源路径：`static/images/providers/atlas-cloud.png`。
- 已补充配置示例：`deployment/.env.example`、`test/provider_testing_stack/.env.test.example`。
- 已补充单元测试，覆盖 `ATLAS` provider 路由和配置校验。

## 已验证内容

- `tests/unit/test_ai.py` 本地通过：46 passed。
- 代码静态诊断通过，已修改文件无新增 diagnostics。
- 本地密钥已写入忽略文件 `.env.atlas.local`，未纳入 git。
- 参考 `/Users/zby/listPrice` 的 Atlas 调用方式后，已确认当前可用组合：
- `base_url=https://api.atlascloud.ai/v1`
- `model=deepseek-ai/DeepSeek-V3-0324`
- 使用当前项目代码路径已验证成功：
- 文本生成返回 `PROJECT_OK`
- tool calling 返回 `echo_payload(message=TOOL_OK)`

## 关键修正

- 之前真实接口探测失败的原因不是路径本身，而是探测时使用了错误模型名。
- 按 `/Users/zby/listPrice` 的现有可用配置改为 `deepseek-ai/DeepSeek-V3-0324` 后，请求成功。

## 下一步

- 当前代码已经具备提交 PR 条件。
- 如需继续扩大验证范围，可以在后续补做页面级联调或整站启动验证，但本次 provider 接入本身已经成功。
