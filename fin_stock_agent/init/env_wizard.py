from __future__ import annotations

from pathlib import Path

import streamlit as st

from fin_stock_agent.core.settings import settings


class EnvWizard:
    def render(self) -> bool:
        st.title("FinStock-Agent Setup")
        with st.form("env_wizard"):
            openai_base_url = st.text_input("OpenAI Base URL", value=settings.openai_base_url)
            openai_api_key = st.text_input("OpenAI API Key", value=settings.openai_api_key, type="password")
            openai_model = st.text_input("OpenAI Model", value=settings.openai_model)
            tushare_token = st.text_input("Tushare Token", value=settings.tushare_token, type="password")
            submitted = st.form_submit_button("Save configuration")
        if not submitted:
            return False
        if not openai_api_key or not tushare_token:
            st.error("OPENAI_API_KEY and TUSHARE_TOKEN are required.")
            return False
        content = "\n".join(
            [
                f"OPENAI_BASE_URL={openai_base_url}",
                f"OPENAI_API_KEY={openai_api_key}",
                f"OPENAI_MODEL={openai_model}",
                f"TUSHARE_TOKEN={tushare_token}",
            ]
        )
        Path(settings.env_file).write_text(content + "\n", encoding="utf-8")
        st.success("Configuration saved. Rerun the app to continue.")
        return True
