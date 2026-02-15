"""
消息通知模块 — 邮件 / 微信 / 钉钉。
"""

from __future__ import annotations

import smtplib
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any

import requests

from stockquant.utils.config import Config
from stockquant.utils.logger import get_logger

logger = get_logger("notify")


class Notifier(ABC):
    """通知基类。"""

    @abstractmethod
    def send(self, title: str, content: str) -> bool:
        """发送通知。"""


class EmailNotifier(Notifier):
    """邮件通知。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self.smtp_server: str = cfg.get("notify.channels.email.smtp_server", "")
        self.smtp_port: int = cfg.get("notify.channels.email.smtp_port", 465)
        self.sender: str = cfg.get("notify.channels.email.sender", "")
        self.password: str = cfg.get("notify.channels.email.password", "")
        self.receivers: list[str] = cfg.get("notify.channels.email.receivers", [])

    def send(self, title: str, content: str) -> bool:
        if not self.smtp_server or not self.sender:
            logger.warning("邮件通知未配置")
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.sender
            msg["To"] = ", ".join(self.receivers)
            msg["Subject"] = title
            msg.attach(MIMEText(content, "html", "utf-8"))

            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.receivers, msg.as_string())

            logger.info(f"邮件发送成功: {title}")
            return True
        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return False


class WechatNotifier(Notifier):
    """企业微信机器人通知。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self.webhook_url: str = cfg.get("notify.channels.wechat.webhook_url", "")

    def send(self, title: str, content: str) -> bool:
        if not self.webhook_url:
            logger.warning("企业微信 webhook 未配置")
            return False

        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {"content": f"## {title}\n{content}"},
            }
            resp = requests.post(self.webhook_url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"微信通知发送成功: {title}")
            return True
        except Exception as e:
            logger.error(f"微信通知发送失败: {e}")
            return False


class DingTalkNotifier(Notifier):
    """钉钉机器人通知。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self.webhook_url: str = cfg.get("notify.channels.dingtalk.webhook_url", "")
        self.secret: str = cfg.get("notify.channels.dingtalk.secret", "")

    def send(self, title: str, content: str) -> bool:
        if not self.webhook_url:
            logger.warning("钉钉 webhook 未配置")
            return False

        try:
            url = self._sign_url() if self.secret else self.webhook_url
            payload = {
                "msgtype": "markdown",
                "markdown": {"title": title, "text": f"## {title}\n{content}"},
            }
            resp = requests.post(url, json=payload, timeout=10)
            resp.raise_for_status()
            logger.info(f"钉钉通知发送成功: {title}")
            return True
        except Exception as e:
            logger.error(f"钉钉通知发送失败: {e}")
            return False

    def _sign_url(self) -> str:
        """钉钉加签。"""
        import hashlib
        import hmac
        import base64
        import time
        import urllib.parse

        timestamp = str(round(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))
        return f"{self.webhook_url}&timestamp={timestamp}&sign={sign}"


class NotifyManager:
    """通知管理器 — 统一调度多渠道通知。"""

    def __init__(self, config: Config | None = None) -> None:
        cfg = config or Config()
        self._channels: list[Notifier] = []

        if cfg.get("notify.enabled", False):
            if cfg.get("notify.channels.email.enabled", False):
                self._channels.append(EmailNotifier(cfg))
            if cfg.get("notify.channels.wechat.enabled", False):
                self._channels.append(WechatNotifier(cfg))
            if cfg.get("notify.channels.dingtalk.enabled", False):
                self._channels.append(DingTalkNotifier(cfg))

    def send(self, title: str, content: str) -> None:
        """向所有启用的渠道发送通知。"""
        for ch in self._channels:
            ch.send(title, content)
