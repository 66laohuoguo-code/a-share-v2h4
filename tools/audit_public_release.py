"""上传前审计：算出"会被 Git 跟踪的文件"，并对文件名/内容做泄露检查。

为什么需要它
------------
`.gitignore` 有两种写法：**枚举**（逐个列出要排除的文件）和**模式**（用通配）。
枚举看起来精确，但它**必然落后**于新增文件 —— 加了一个新模块却忘了加一行，
下次 `git add .` 就把它带上去了。本脚本的作用是在 `git add` **之前**把这件事变成
可执行检查，让仓库能自我保护。

本脚本本身不含任何机器特定信息：需要排除的路径片段由 `--forbid` 传入。

用法
----
    python tools/audit_public_release.py --root .
    python tools/audit_public_release.py --staged --forbid "D:\\data" --forbid "myuser"

`--staged` 时读取 `.git/index` 里已暂存的文件（需要 `git` 可用）；
否则按 `.gitignore` 语义计算"`git add .` 会加进去哪些文件"。
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".mypy_cache",
             ".ruff_cache", ".idea", ".vscode"}

# 默认拦截的**文件名**模式（不是内容）：这些名字本身就是"研究中间产物"
DEFAULT_FORBIDDEN_NAMES = [
    "*_factor_research.py", "*_factor_research.ps1", "*_LOCAL.md",
    "config/*factor*.json", "*.sqlite", "*.sqlite-*", "*.db", "*.parquet",
    "*.log", "*.zip", "resume/*", "*_private*", "*.pem", "*.key",
]
# 默认拦截的**内容**模式：密钥与凭据（不含任何机器特定路径）
DEFAULT_FORBIDDEN_CONTENT = [
    (r"\bsk-[A-Za-z0-9]{16,}", "疑似 API Key"),
    (r"\bAKIA[0-9A-Z]{16}\b", "疑似 AWS Access Key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "私钥"),
    (r"(?i)\b(api[_-]?key|secret|passwd|password|token)\s*[:=]\s*['\"][^'\"]{8,}",
     "疑似凭据赋值"),
    (r"\bghp_[A-Za-z0-9]{20,}", "疑似 GitHub Token"),
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".ps1", ".sh",
                 ".cfg", ".ini", ".toml", ".csv", ".sql", ".tex", ".example"}


def load_gitignore(path: str) -> list[tuple[str, bool, bool]]:
    rules: list[tuple[str, bool, bool]] = []
    if not os.path.exists(path):
        return rules
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            dir_only = line.endswith("/")
            line = line.strip().strip("/")
            if line:
                rules.append((line, negated, dir_only))
    return rules


def _match(pattern: str, path: str) -> bool:
    if "/" not in pattern:
        return fnmatch.fnmatch(os.path.basename(path), pattern)
    left, right = pattern.split("/"), path.split("/")
    if len(left) != len(right):
        return False
    return all(fnmatch.fnmatch(a, b) for a, b in zip(right, left))


def _decide(rules, path: str, is_dir: bool) -> bool:
    state = False
    for pattern, negated, dir_only in rules:
        if dir_only and not is_dir:
            continue
        if _match(pattern, path):
            state = not negated
    return state


def is_ignored(rules, relpath: str) -> bool:
    parts = relpath.split("/")
    for index in range(1, len(parts)):            # 祖先目录被排除则无法用 ! 救回
        if _decide(rules, "/".join(parts[:index]), True):
            return True
    return _decide(rules, relpath, False)


def collect(root: str, rules) -> list[str]:
    staged = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in files:
            rel = os.path.relpath(os.path.join(base, name), root).replace("\\", "/")
            if not is_ignored(rules, rel):
                staged.append(rel)
    return sorted(staged)


def read_staged(root: str) -> list[str]:
    """从 .git/index 读已暂存路径（无 git 命令时的兜底：解 index 太重，这里直接提示）。"""
    raise SystemExit("--staged 需要 `git diff --cached --name-only`；"
                     "请先运行它并把输出用 --from-file 传入。可改用默认模式。")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="公开上传前审计")
    parser.add_argument("--root", default=".")
    parser.add_argument("--from-file", default=None,
                        help="要检查的文件清单（每行一个）；省略则按 .gitignore 计算")
    parser.add_argument("--forbid", action="append", default=[],
                        help="额外禁止的路径片段（可重复），例如本机目录名或用户名")
    parser.add_argument("--max-bytes", type=int, default=2_000_000)
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as handle:
            files = [line.strip() for line in handle if line.strip()]
    else:
        files = collect(root, load_gitignore(os.path.join(root, ".gitignore")))

    forbidden_names = DEFAULT_FORBIDDEN_NAMES + [f"*{f}*" for f in args.forbid]
    content_rules = list(DEFAULT_FORBIDDEN_CONTENT)
    for fragment in args.forbid:
        content_rules.append((re.escape(fragment), "用户指定的敏感片段"))

    print("待检查文件：%d 个" % len(files))
    name_hits, content_hits = [], []
    for rel in files:
        for pattern in forbidden_names:
            if _match(pattern.lower(), rel.lower()):
                name_hits.append((rel, pattern))
                break
        full = os.path.join(root, rel)
        if os.path.splitext(rel)[1].lower() not in TEXT_SUFFIXES:
            continue
        try:
            if os.path.getsize(full) > args.max_bytes:
                continue
            with open(full, encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
        except OSError:
            continue
        for pattern, label in content_rules:
            found = re.findall(pattern, text)
            if found:
                content_hits.append((rel, label, len(found)))
                break

    print("\n" + "=" * 88)
    print("A) 文件名命中禁止模式：%d 个" % len(name_hits))
    for rel, pattern in name_hits:
        print("   ⚠ %-64s [%s]" % (rel, pattern))

    print("\nB) 文件内容命中敏感模式：%d 个" % len(content_hits))
    for rel, label, count in content_hits:
        print("   ⚠ %-64s [%s ×%d]" % (rel, label, count))

    if name_hits or content_hits:
        print("\n结论：**不要推送**。先修 .gitignore 或把命中文件从暂存区移除：")
        print("   git restore --staged <文件>")
        print("并把规则写进 .gitignore（优先用**模式**而不是逐条枚举）。")
        return 1
    print("\n结论：未发现命中，可以推送。")
    print("提示：推送前仍然执行 `git diff --cached --name-only` 人工过一遍。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
