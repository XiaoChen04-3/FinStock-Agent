from __future__ import annotations

import streamlit as st

from fin_stock_agent.core.settings import settings
from fin_stock_agent.init.data_preloader import DataPreloader
from fin_stock_agent.init.env_wizard import EnvWizard
from fin_stock_agent.storage.database import init_db


class SystemInit:
    def __init__(self) -> None:
        self.preloader = DataPreloader()
        self.wizard = EnvWizard()

    def check_and_setup(self) -> bool:
        init_db()
        if not settings.is_configured():
            return self.wizard.render()
        if "system_preloaded" not in st.session_state:
            self.preloader.preload()
            st.session_state["system_preloaded"] = True
        return True
