from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from fin_stock_agent.core.config import get_config
from fin_stock_agent.core.settings import settings

logger = logging.getLogger(__name__)

DEFAULT_USER_PROFILE_MD = """# 用户画像

## 投资偏好
- 风险承受：未知
- 投资期限：未知
- 偏好资产：暂无
- 规避事项：暂无

## 关注范围
- 关注主题：暂无
- 自选标的：暂无

## 回答偏好
- 暂无

## 决策约束
- 仅作研究参考，不替用户下确定性买卖指令
"""

_DANGEROUS_PATTERNS = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "忽略系统提示",
    "忽略所有系统",
    "越过系统",
    "api_key",
    "openai_api_key",
    "tushare_token",
    "password",
    "passwd",
    "secret_key",
    "<script",
    "</script",
)


@dataclass(frozen=True)
class UserProfileSnapshot:
    active_text: str
    staged_text: str
    path: Path
    pending_path: Path
    token_estimate: int
    pending_exists: bool


class UserProfileValidationError(ValueError):
    """用户画像 Markdown 不安全或超过预算时抛出。"""


class UserProfileFileService:
    """基于文件的用户画像记忆，支持运行期冻结与 pending 提交。

    active 画像只在初始化时读取一次，并在当前进程生命周期内保持稳定。
    每轮对话提取的新画像只写入 pending 文件；提交时再原子替换正式文件。
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._initialized = False
        self._shutdown_registered = False
        self._active_text = ""
        self._staged_text = ""
        self._path: Path | None = None
        self._pending_path: Path | None = None
        self._backup_path: Path | None = None

    def initialize(self, *, initial_text: str | None = None, force: bool = False) -> None:
        with self._lock:
            if self._initialized and not force:
                return
            cfg = get_config().memory.user_profile
            self._path = self._resolve_path(cfg.path)
            self._pending_path = self._resolve_path(cfg.pending_path)
            self._backup_path = self._path.with_suffix(self._path.suffix + ".backup")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            self._fallback_if_unwritable_locked()

            if cfg.commit_on_shutdown and self._pending_path.exists():
                self._recover_pending_locked()

            if not self._path.exists() or not self._path.read_text(encoding="utf-8").strip():
                seed = self.normalize_markdown(initial_text or DEFAULT_USER_PROFILE_MD)
                self.validate_markdown(seed)
                self._atomic_write(self._path, seed)

            active = self.normalize_markdown(self._path.read_text(encoding="utf-8"))
            try:
                self.validate_markdown(active)
            except UserProfileValidationError as exc:
                invalid_path = self._path.with_suffix(self._path.suffix + ".invalid")
                shutil.move(str(self._path), str(invalid_path))
                logger.warning("用户画像文件校验失败，已移动到 %s：%s", invalid_path, exc)
                active = DEFAULT_USER_PROFILE_MD
                self._atomic_write(self._path, active)

            self._active_text = active
            self._staged_text = active
            self._initialized = True
            self.register_shutdown_handler()

    def register_shutdown_handler(self) -> None:
        with self._lock:
            if self._shutdown_registered:
                return
            atexit.register(self._shutdown_commit)
            self._shutdown_registered = True

    def get_active_profile_text(self) -> str:
        self.initialize()
        with self._lock:
            return self._active_text

    def get_staged_profile_text(self) -> str:
        self.initialize()
        with self._lock:
            return self._staged_text

    def stage_profile(self, profile_md: str) -> bool:
        self.initialize()
        normalized = self.normalize_markdown(profile_md)
        self.validate_markdown(normalized)
        with self._lock:
            if normalized == self._staged_text:
                return False
            self._atomic_write(self._pending_path_required(), normalized)
            self._staged_text = normalized
            return True

    def commit_pending(self) -> bool:
        self.initialize()
        with self._lock:
            pending_path = self._pending_path_required()
            if not pending_path.exists():
                return False
            pending = self.normalize_markdown(pending_path.read_text(encoding="utf-8"))
            self.validate_markdown(pending)
            if pending == self._active_text:
                pending_path.unlink(missing_ok=True)
                return False
            path = self._path_required()
            backup_path = self._backup_path_required()
            if path.exists():
                shutil.copy2(path, backup_path)
            os.replace(pending_path, path)
            self._active_text = pending
            self._staged_text = pending
            return True

    def reset(self, profile_md: str | None = None) -> None:
        with self._lock:
            cfg = get_config().memory.user_profile
            self._path = self._resolve_path(cfg.path)
            self._pending_path = self._resolve_path(cfg.pending_path)
            self._backup_path = self._path.with_suffix(self._path.suffix + ".backup")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            self._fallback_if_unwritable_locked()
            text = self.normalize_markdown(profile_md or DEFAULT_USER_PROFILE_MD)
            self.validate_markdown(text)
            self._atomic_write(self._path, text)
            self._pending_path.unlink(missing_ok=True)
            self._active_text = text
            self._staged_text = text
            self._initialized = True
            self.register_shutdown_handler()

    def snapshot(self) -> UserProfileSnapshot:
        self.initialize()
        with self._lock:
            active = self._active_text
            staged = self._staged_text
            pending_path = self._pending_path_required()
            return UserProfileSnapshot(
                active_text=active,
                staged_text=staged,
                path=self._path_required(),
                pending_path=pending_path,
                token_estimate=self.estimate_tokens(active),
                pending_exists=pending_path.exists(),
            )

    def validate_markdown(self, profile_md: str) -> None:
        text = self.normalize_markdown(profile_md)
        if not text.startswith("# 用户画像"):
            raise UserProfileValidationError("用户画像 Markdown 必须以 '# 用户画像' 开头")
        if "```" in text:
            raise UserProfileValidationError("用户画像 Markdown 不允许包含代码块")
        lowered = text.lower()
        for pattern in _DANGEROUS_PATTERNS:
            if pattern in lowered or pattern in text:
                raise UserProfileValidationError(f"用户画像包含不安全短语：{pattern}")
        max_tokens = get_config().memory.user_profile.max_tokens
        tokens = self.estimate_tokens(text)
        if tokens > max_tokens:
            raise UserProfileValidationError(
                f"用户画像超过 token 预算：预估 {tokens}，上限 {max_tokens}"
            )

    @staticmethod
    def normalize_markdown(profile_md: str) -> str:
        text = (profile_md or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            return DEFAULT_USER_PROFILE_MD.strip() + "\n"
        if not text.startswith("# 用户画像"):
            text = "# 用户画像\n\n" + text.lstrip("# \n")
        return text.rstrip() + "\n"

    @staticmethod
    def estimate_tokens(text: str) -> int:
        # 对中英文混合金融画像做偏保守的 token 估算，确保不会突破配置预算。
        chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
        ascii_words = len(re.findall(r"[A-Za-z0-9_.%+-]+", text or ""))
        punctuation = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text or "")) // 4
        return chinese_chars + ascii_words + punctuation

    def _shutdown_commit(self) -> None:
        try:
            try:
                from fin_stock_agent.agents.router import flush_post_turn_tasks

                flush_post_turn_tasks(timeout=15.0)
            except Exception as exc:
                logger.warning("用户画像关闭提交：等待对话后任务完成失败：%s", exc)
            if get_config().memory.user_profile.commit_on_shutdown:
                self.commit_pending()
        except Exception as exc:
            logger.warning("用户画像关闭提交失败：%s", exc)

    def _recover_pending_locked(self) -> None:
        pending_path = self._pending_path_required()
        if not pending_path.exists():
            return
        try:
            pending = self.normalize_markdown(pending_path.read_text(encoding="utf-8"))
            self.validate_markdown(pending)
            path = self._path_required()
            if path.exists():
                shutil.copy2(path, self._backup_path_required())
            os.replace(pending_path, path)
            logger.info("已恢复 pending 用户画像到 %s", path)
        except Exception as exc:
            invalid_path = pending_path.with_suffix(pending_path.suffix + ".invalid")
            try:
                os.replace(pending_path, invalid_path)
            except Exception:
                logger.exception("隔离无效 pending 用户画像失败")
            logger.warning("无效 pending 用户画像已隔离到 %s：%s", invalid_path, exc)

    def _resolve_path(self, raw: str) -> Path:
        root = settings.project_root.resolve()
        path = Path(raw)
        resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise UserProfileValidationError(f"用户画像路径必须位于项目根目录下：{raw}") from exc
        return resolved

    def _atomic_write(self, path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def _fallback_if_unwritable_locked(self) -> None:
        path = self._path_required()
        pending_path = self._pending_path_required()
        probe = pending_path.with_suffix(pending_path.suffix + ".probe")
        try:
            probe.parent.mkdir(parents=True, exist_ok=True)
            with probe.open("w", encoding="utf-8") as handle:
                handle.write("ok")
            probe.unlink(missing_ok=True)
            return
        except OSError as exc:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass
            fallback_path = settings.project_root / "user_runtime.md"
            fallback_pending = settings.project_root / "user_runtime.pending.md"
            logger.warning(
                "配置的用户画像路径不可写（%s），已临时降级到 %s",
                exc,
                fallback_path,
            )
            if path.exists() and not fallback_path.exists():
                try:
                    shutil.copy2(path, fallback_path)
                except OSError:
                    pass
            self._path = fallback_path
            self._pending_path = fallback_pending
            self._backup_path = fallback_path.with_suffix(fallback_path.suffix + ".backup")

    def _path_required(self) -> Path:
        if self._path is None:
            self._path = self._resolve_path(get_config().memory.user_profile.path)
        return self._path

    def _pending_path_required(self) -> Path:
        if self._pending_path is None:
            self._pending_path = self._resolve_path(get_config().memory.user_profile.pending_path)
        return self._pending_path

    def _backup_path_required(self) -> Path:
        if self._backup_path is None:
            self._backup_path = self._path_required().with_suffix(self._path_required().suffix + ".backup")
        return self._backup_path


_USER_PROFILE_FILE_SERVICE = UserProfileFileService()


def get_user_profile_file_service() -> UserProfileFileService:
    return _USER_PROFILE_FILE_SERVICE


def reset_user_profile_file_service_for_tests() -> None:
    global _USER_PROFILE_FILE_SERVICE
    _USER_PROFILE_FILE_SERVICE = UserProfileFileService()
